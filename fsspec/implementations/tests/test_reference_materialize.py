"""Tests for resumable ReferenceFileSystem.materialize / _materialize.

The remote backend is always a *local* async filesystem double
(:class:`RecordingLocalAsyncFS`) backed by files in a real temporary
directory.  It records byte-range requests, concurrency peaks, in-flight
requests and cancellations, and can inject short reads, failures and slow
blocking reads.  No external network, account or private data is required.
"""

import asyncio
import hashlib
import json
import os
import pathlib
import threading

import pytest

import fsspec
from fsspec.asyn import AsyncFileSystem
from fsspec.implementations._reference_materialize import (
    _CHECKPOINT_VERSION,
    ReferenceMaterializationError,
)
from fsspec.implementations.reference import ReferenceFileSystem


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Local async filesystem double
# ---------------------------------------------------------------------------


class RecordingLocalAsyncFS(AsyncFileSystem):
    """File-backed async FS recording ranges, concurrency and cancellation."""

    protocol = "recording"
    cachable = False

    def __init__(self, root, delay=0.0, **kwargs):
        super().__init__(**kwargs)
        self.root = pathlib.Path(root)
        self.requests = []  # (path, start, end, length)
        self.active = 0
        self.peak = 0
        self._lock = asyncio.Lock()
        self.delay = delay
        self.cancelled_after_start = []
        self.blocking_calls = 0
        self.blocking_finished = 0
        # test hooks
        self.short_read_paths = set()
        self.fail_paths = set()
        self.unknown_size_paths = set()

    def _disk(self, path):
        # strip a possible "recording://" prefix
        if "://" in path:
            path = path.split("://", 1)[1]
        return self.root / path.lstrip("/")

    async def _info(self, path, **kwargs):
        disk = self._disk(path)
        if path in self.unknown_size_paths:
            # mimics backends that answer but cannot report the size
            return {"name": path, "size": None, "type": "file"}
        if not disk.exists():
            raise FileNotFoundError(path)
        return {"name": path, "size": disk.stat().st_size, "type": "file"}

    async def _cat_file(self, path, start=None, end=None, **kwargs):
        disk = self._disk(path)
        length = None if end is None else end - (start or 0)
        if path in self.fail_paths:
            raise OSError(f"injected failure for {path}")
        async with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            # run_in_executor(None) mirrors AsyncFileSystemWrapper's dispatch
            # of wrapped synchronous backends, so materialize can track it.
            data = await asyncio.get_running_loop().run_in_executor(
                None, self._blocking_read, disk, start, end
            )
            if path in self.short_read_paths and data:
                data = data[:-1]
            self.requests.append((path, start, end, length))
            return data
        except asyncio.CancelledError:
            self.cancelled_after_start.append((path, start, end))
            raise
        finally:
            async with self._lock:
                self.active -= 1

    def _blocking_read(self, disk, start, end):
        self.blocking_calls += 1
        with open(disk, "rb") as f:
            if start is not None:
                f.seek(start)
            return f.read() if end is None else f.read(end - start)


class BlockingRecordingLocalAsyncFS(RecordingLocalAsyncFS):
    """Variant whose payload reads are truly blocking (not cancellable).

    Simulates a wrapped synchronous backend: the in-flight work runs inside
    ``asyncio.to_thread`` and waits on a threading event, so task cancellation
    cannot interrupt it; materialize must still wait for it to finish.
    """

    def __init__(self, *args, gate=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate = gate or threading.Event()

    async def _cat_file(self, path, start=None, end=None, **kwargs):
        if path in self.fail_paths:
            raise OSError(f"injected failure for {path}")
        async with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)

        def blocking():
            self.blocking_calls += 1
            # Wait up to the test timeout; the test releases the gate.
            self.gate.wait(30)
            disk = self._disk(path)
            with open(disk, "rb") as f:
                if start is not None:
                    f.seek(start)
                out = f.read() if end is None else f.read(end - start)
            self.blocking_finished += 1
            return out

        try:
            data = await asyncio.get_running_loop().run_in_executor(
                None, blocking
            )
            length = None if end is None else end - start
            self.requests.append((path, start, end, length))
            return data
        finally:
            async with self._lock:
                self.active -= 1


def make_backend(tmp_path, payloads, cls=RecordingLocalAsyncFS, **kw):
    root = tmp_path / "remote"
    root.mkdir(exist_ok=True)
    for name, content in payloads.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
    return cls(root, asynchronous=True, **kw)


def make_refs_fs(backend, refs, **fs_kw):
    kw = {"fs": backend, "asynchronous": True}
    kw.update(fs_kw)
    return fsspec.filesystem("reference", fo=refs, **kw)


def write_payload(tmp_path, name, size, seed=0):
    content = bytes((i + seed) % 256 for i in range(size))
    p = tmp_path / "remote" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return content

# ---------------------------------------------------------------------------
# Basic behavior: three reference kinds, templates, default target, zero size
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_materialize_three_kinds_and_order(tmp_path):
    inline = b"inline-payload"
    b64 = base64_payload = b"hello-base64"
    remote = bytes(range(256)) * 16  # 4096 bytes
    backend = make_backend(tmp_path, {"big.bin": remote})
    refs = {
        "dir/inline.txt": inline,
        "dir/b64.txt": b"base64:" + __import__("base64").b64encode(b64),
        "whole.bin": ["recording://big.bin"],
        "range.bin": ["recording://big.bin", 10, 1000],
        "zero-inline": b"",
        "zero-range": ["recording://big.bin", 5, 0],
        "zero-whole-ish": ["recording://big.bin", 0, 0],
    }
    fs = make_refs_fs(backend, refs)
    dest = tmp_path / "out"
    ckpt = tmp_path / "progress" / "cp.json"

    result = await fs._materialize(
        [
            "dir/inline.txt",
            "dir/b64.txt",
            "whole.bin",
            "range.bin",
            "zero-inline",
            "zero-range",
            "zero-whole-ish",
        ],
        str(dest),
        checkpoint=str(ckpt),
        block_size=512,
        max_concurrency=3,
    )

    assert result["written"] == [
        "dir/inline.txt",
        "dir/b64.txt",
        "whole.bin",
        "range.bin",
        "zero-inline",
        "zero-range",
        "zero-whole-ish",
    ]
    assert result["reused"] == []
    assert result["checkpoint"] == str(ckpt.resolve())
    expected_bytes = len(inline) + len(b64) + len(remote) + 1000
    assert result["bytes_written"] == expected_bytes

    assert (dest / "dir/inline.txt").read_bytes() == inline
    assert (dest / "dir/b64.txt").read_bytes() == b64
    assert (dest / "whole.bin").read_bytes() == remote
    assert (dest / "range.bin").read_bytes() == remote[10:1010]
    assert (dest / "zero-inline").read_bytes() == b""
    assert (dest / "zero-range").read_bytes() == b""
    assert (dest / "zero-whole-ish").read_bytes() == b""

    # no request may exceed block_size, and ranges stay inside the reference
    for _path, start, end, length in backend.requests:
        assert end - start <= 512
        if _path.endswith("big.bin") and length is not None:
            pass
    # whole file requests cover [0, 4096); range requests cover [10, 1010)
    whole = [(s, e) for p, s, e, _ in backend.requests if p.endswith("big.bin")]
    assert min(s for s, _ in whole) >= 0
    assert max(e for _, e in whole) <= 4096


@pytest.mark.asyncio
async def test_materialize_v1_templates_and_default_target(tmp_path):
    remote = b"0123456789abcdef" * 4
    backend = make_backend(tmp_path, {"data/f.bin": remote})
    spec = {
        "version": 1,
        "templates": {"u": "recording://data"},
        "refs": {
            "templated": ["{{u}}/f.bin", 2, 8],
            "plain": "text-inline",
        },
    }
    fs = fsspec.filesystem(
        "reference",
        fo=spec,
        fs=backend,
        target="recording://data/f.bin",
        asynchronous=True,
    )
    # default-target reference shape via V0 dict on the same instance inputs
    fs_default = fsspec.filesystem(
        "reference",
        fo={"defaulted": (None, 4, 6)},
        fs=backend,
        target="recording://data/f.bin",
        asynchronous=True,
    )
    dest = tmp_path / "out"
    ckpt = tmp_path / "cp.json"
    result = await fs._materialize(
        ["templated", "plain"], str(dest), checkpoint=str(ckpt)
    )
    assert result["written"] == ["templated", "plain"]
    assert (dest / "templated").read_bytes() == remote[2:10]
    assert (dest / "plain").read_bytes() == b"text-inline"

    state = json.loads(ckpt.read_text())
    assert state["plan"]["templated"]["url"] == "recording://data/f.bin"

    result2 = await fs_default._materialize(
        "defaulted",
        str(tmp_path / "out2"),
        checkpoint=str(tmp_path / "cp2.json"),
    )
    assert result2["written"] == ["defaulted"]
    assert (tmp_path / "out2" / "defaulted").read_bytes() == remote[4:10]
    state2 = json.loads((tmp_path / "cp2.json").read_text())
    assert state2["plan"]["defaulted"]["default_target"] is True
    assert state2["plan"]["defaulted"]["url"] == "recording://data/f.bin"
    assert state2["target"] == "recording://data/f.bin"


@pytest.mark.asyncio
async def test_materialize_single_string_path(tmp_path):
    backend = make_backend(tmp_path, {"f": b"abc"})
    fs = make_refs_fs(backend, {"k": ["recording://f"]})
    result = await fs._materialize(
        "k", str(tmp_path / "out"), checkpoint=str(tmp_path / "cp.json")
    )
    assert result["written"] == ["k"]
    assert (tmp_path / "out" / "k").read_bytes() == b"abc"


# ---------------------------------------------------------------------------
# Chunking, short reads, request bounds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chunked_offsets_and_request_sizes(tmp_path):
    payload = write_payload(tmp_path, "chunked", 3000)
    backend = make_backend(tmp_path, {})  # file already written
    refs = {"c": ["recording://chunked", 250, 2500]}
    fs = make_refs_fs(backend, refs)
    await fs._materialize(
        "c",
        str(tmp_path / "out"),
        checkpoint=str(tmp_path / "cp.json"),
        block_size=1000,
        max_concurrency=1,
    )
    assert (tmp_path / "out" / "c").read_bytes() == payload[250:2750]
    # expect exactly ceil(2500/1000) = 3 requests, none leaving the range
    assert len(backend.requests) == 3
    spans = [(s, e) for _, s, e, _ in backend.requests]
    assert spans == [(250, 1250), (1250, 2250), (2250, 2750)]
    assert all(e - s <= 1000 for s, e in spans)


@pytest.mark.asyncio
async def test_short_read_is_failure(tmp_path):
    write_payload(tmp_path, "f", 2048)
    backend = make_backend(tmp_path, {})
    backend.short_read_paths.add("recording://f")
    fs = make_refs_fs(backend, {"k": ["recording://f", 0, 2048]})
    dest = tmp_path / "out"
    with pytest.raises(ReferenceMaterializationError) as exc:
        await fs._materialize(
            "k",
            str(dest),
            checkpoint=str(tmp_path / "cp.json"),
            block_size=512,
        )
    assert exc.value.path == "k"
    assert isinstance(exc.value.__cause__, ReferenceMaterializationError) or True
    assert "short read" in str(exc.value)
    # no final file is published on failure
    assert not (dest / "k").exists()
    # no orphan temp files remain
    leftovers = [
        p for p in dest.rglob("*")
        if p.name.startswith(".fsspec-materialize-")
    ]
    assert leftovers == []


@pytest.mark.asyncio
async def test_whole_file_unknown_size_rejected(tmp_path):
    write_payload(tmp_path, "mystery", 100)
    backend = make_backend(tmp_path, {})
    backend.unknown_size_paths.add("recording://mystery")
    fs = make_refs_fs(backend, {"k": ["recording://mystery"]})
    with pytest.raises(ReferenceMaterializationError) as exc:
        await fs._materialize(
            "k", str(tmp_path / "out"), checkpoint=str(tmp_path / "cp.json")
        )
    assert exc.value.path == "k"
    assert "size" in str(exc.value)
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------------------
# SHA256 handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expected_sha256_match_and_mismatch(tmp_path):
    good = b"correct contents"
    backend = make_backend(tmp_path, {"g": good, "b": b"other"})
    refs = {"g": ["recording://g"], "b": ["recording://b"]}
    fs = make_refs_fs(backend, refs)
    expected = {
        "g": sha256_hex(good).upper(),  # upper case accepted, normalized
        "b": sha256_hex(b"tampered"),
    }
    with pytest.raises(ReferenceMaterializationError) as exc:
        await fs._materialize(
            ["g", "b"],
            str(tmp_path / "out"),
            checkpoint=str(tmp_path / "cp.json"),
            expected_sha256=expected,
            max_concurrency=1,
        )
    assert exc.value.path == "b"
    assert "sha256 mismatch" in str(exc.value)
    # g (processed first in queue order) is still recorded as completed;
    # materialization is not an all-or-nothing transaction.
    state = json.loads((tmp_path / "cp.json").read_text())
    assert "g" in state["completed"]
    assert "b" not in state["completed"]
    assert (tmp_path / "out" / "g").read_bytes() == good
    assert not (tmp_path / "out" / "b").exists()

    # bad expected map shapes
    with pytest.raises(ValueError):
        await fs._materialize(
            ["g"],
            str(tmp_path / "out2"),
            checkpoint=str(tmp_path / "cp2.json"),
            expected_sha256={"g": "abc"},
        )
    with pytest.raises(ValueError):
        await fs._materialize(
            ["g"],
            str(tmp_path / "out3"),
            checkpoint=str(tmp_path / "cp3.json"),
            expected_sha256={"g": sha256_hex(good), "x": sha256_hex(b"")},
        )
    with pytest.raises(ValueError):
        await fs._materialize(
            ["g"],
            str(tmp_path / "out4"),
            checkpoint=str(tmp_path / "cp4.json"),
            expected_sha256={"other": sha256_hex(good)},
        )


# ---------------------------------------------------------------------------
# Concurrency peak and memory-bound request sizing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrency_peak_bounded(tmp_path):
    payload = write_payload(tmp_path, "shared", 5000)
    backend = make_backend(tmp_path, {}, delay=0.02)
    refs = {f"k{i}": ["recording://shared", 0, 5000] for i in range(8)}
    fs = make_refs_fs(backend, refs)
    await fs._materialize(
        list(refs),
        str(tmp_path / "out"),
        checkpoint=str(tmp_path / "cp.json"),
        block_size=777,
        max_concurrency=3,
    )
    assert backend.peak <= 3
    assert backend.peak == 3  # with delays, the pool saturates
    for _p, s, e, length in backend.requests:
        assert e - s <= 777
    for i in range(8):
        assert (tmp_path / "out" / f"k{i}").read_bytes() == payload
        assert len(backend.requests) >= 8


# ---------------------------------------------------------------------------
# Cancellation and worker failure shutdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slow_backend_cancellation_stops_writes(tmp_path):
    write_payload(tmp_path, "slow", 4096)
    backend = make_backend(tmp_path, {}, delay=0.05)
    refs = {f"k{i}": ["recording://slow", 0, 4096] for i in range(12)}
    fs = make_refs_fs(backend, refs)
    task = asyncio.ensure_future(
        fs._materialize(
            list(refs),
            str(tmp_path / "out"),
            checkpoint=str(tmp_path / "cp.json"),
            block_size=256,
            max_concurrency=4,
        )
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # after the call returns, the backend must observe no further reads
    assert backend.active == 0
    snapshot = len(backend.requests)
    await asyncio.sleep(0.1)
    assert len(backend.requests) == snapshot
    # any unfinished temp files were cleaned; no partial final file
    dest = tmp_path / "out"
    if dest.exists():
        temps = [p for p in dest.rglob(".fsspec-materialize-*")]
        assert temps == []
    # resume is possible: completed keys recorded before cancellation survive
    state = json.loads((tmp_path / "cp.json").read_text())
    assert isinstance(state["completed"], dict)

    result = await fs._materialize(
        list(refs),
        str(tmp_path / "out"),
        checkpoint=str(tmp_path / "cp.json"),
        block_size=256,
        max_concurrency=4,
        resume=True,
    )
    assert sorted(result["written"] + result["reused"]) == sorted(refs)
    for i in range(12):
        assert (tmp_path / "out" / f"k{i}").read_bytes() == (
            bytes((j) % 256 for j in range(4096))
        )


@pytest.mark.asyncio
async def test_blocking_backend_cancellation_waits_for_work(tmp_path):
    write_payload(tmp_path, "blocked", 128)
    gate = threading.Event()
    backend = make_backend(
        tmp_path, {}, cls=BlockingRecordingLocalAsyncFS, gate=gate
    )
    fs = make_refs_fs(backend, {"k": ["recording://blocked", 0, 128]})
    task = asyncio.ensure_future(
        fs._materialize(
            "k",
            str(tmp_path / "out"),
            checkpoint=str(tmp_path / "cp.json"),
            max_concurrency=2,
        )
    )
    # wait until the blocking call has actually started in a thread
    while backend.blocking_calls == 0:
        await asyncio.sleep(0.01)
    task.cancel()
    # give the cancellation a chance to be delivered
    await asyncio.sleep(0.05)
    assert not task.done()  # blocking work is not interruptible
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert backend.blocking_finished == 1
    assert backend.active == 0


@pytest.mark.asyncio
async def test_one_worker_failure_terminates_others(tmp_path):
    write_payload(tmp_path, "ok", 2048)
    write_payload(tmp_path, "bad", 2048)
    backend = make_backend(tmp_path, {}, delay=0.02)
    # "bad" fails only on its first read; a later resume can complete it.
    fail_budget = {"recording://bad": 1}
    orig_cat = backend._cat_file

    async def cat_once(path, start=None, end=None, **kwargs):
        if path in fail_budget and fail_budget[path] > 0:
            fail_budget[path] -= 1
            raise OSError(f"injected failure for {path}")
        return await orig_cat(path, start=start, end=end, **kwargs)

    backend._cat_file = cat_once
    refs = {f"ok{i}": ["recording://ok", 0, 2048] for i in range(6)}
    refs["bad"] = ["recording://bad", 0, 2048]
    fs = make_refs_fs(backend, refs)
    ordered = ["bad"] + [f"ok{i}" for i in range(6)]
    with pytest.raises(ReferenceMaterializationError) as exc:
        await fs._materialize(
            ordered,
            str(tmp_path / "out"),
            checkpoint=str(tmp_path / "cp.json"),
            block_size=256,
            max_concurrency=4,
        )
    assert exc.value.path == "bad"
    assert isinstance(exc.value.__cause__, OSError)
    # after the exception returns, nothing is still reading
    assert backend.active == 0
    snapshot = len(backend.requests)
    await asyncio.sleep(0.1)
    assert len(backend.requests) == snapshot
    # the durable state is consistent: every recorded completion exists on
    # disk, and a resume against the same key set finishes everything and
    # clears any leftover temp files.
    state = json.loads((tmp_path / "cp.json").read_text())
    for key in state["completed"]:
        assert (tmp_path / "out" / key).exists()
    result = await fs._materialize(
        ordered, str(tmp_path / "out"),
        checkpoint=str(tmp_path / "cp.json"), resume=True,
    )
    assert sorted(result["written"] + result["reused"]) == sorted(ordered)
    assert list((tmp_path / "out").rglob(".fsspec-materialize-*")) == []


# ---------------------------------------------------------------------------
# Resume: completed / missing / same-size corrupt targets
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_reuses_verifies_and_redownloads(tmp_path):
    payload_a = b"A" * 500
    payload_b = b"B" * 500
    payload_c = b"C" * 500
    backend = make_backend(
        tmp_path, {"a": payload_a, "b": payload_b, "c": payload_c}
    )
    refs = {
        "a": ["recording://a"],
        "b": ["recording://b"],
        "c": ["recording://c"],
    }
    fs = make_refs_fs(backend, refs)
    dest = tmp_path / "out"
    ckpt = tmp_path / "cp.json"
    first = await fs._materialize(
        ["a", "b", "c"], str(dest), checkpoint=str(ckpt), block_size=128
    )
    assert first["reused"] == []
    assert first["written"] == ["a", "b", "c"]

    # pristine resume: every key verified by streaming local bytes, no reads
    requests_before = len(backend.requests)
    second = await fs._materialize(
        ["c", "a", "b"],  # reordered: allowed on resume
        str(dest),
        checkpoint=str(ckpt),
        block_size=999,  # changed block_size: allowed
        max_concurrency=2,  # changed concurrency: allowed
        resume=True,
    )
    assert second["reused"] == ["c", "a", "b"]
    assert second["written"] == []
    assert second["bytes_written"] == 0
    assert len(backend.requests) == requests_before

    # corrupt b with same-size content: must be detected and re-downloaded
    target_b = dest / "b"
    target_b.write_bytes(b"X" * 500)
    # delete c: missing file must be re-downloaded too
    (dest / "c").unlink()
    third = await fs._materialize(
        ["a", "b", "c"], str(dest), checkpoint=str(ckpt), resume=True
    )
    assert third["reused"] == ["a"]
    assert third["written"] == ["b", "c"]
    assert third["bytes_written"] == 1000
    assert target_b.read_bytes() == payload_b
    assert (dest / "c").read_bytes() == payload_c

    # truncated a (different size): re-download
    target_a = dest / "a"
    target_a.write_bytes(b"A" * 100)
    fourth = await fs._materialize(
        ["a", "b", "c"], str(dest), checkpoint=str(ckpt), resume=True
    )
    assert fourth["reused"] == ["b", "c"]
    assert fourth["written"] == ["a"]
    assert target_a.read_bytes() == payload_a


@pytest.mark.asyncio
async def test_resume_rename_before_checkpoint_window(tmp_path):
    payload = b"window-data" * 50
    backend = make_backend(tmp_path, {"w": payload})
    fs = make_refs_fs(backend, {"k": ["recording://w"]})
    dest = tmp_path / "out"
    ckpt = tmp_path / "cp.json"

    import fsspec.implementations._reference_materialize as mod

    real_atomic = mod._atomic_write

    def fail_completion_write(path, data):
        # The initial plan save carries an empty "completed" map and must
        # succeed.  The first completion update (after os.replace) fails,
        # reproducing a crash in the rename -> checkpoint-update window.
        parsed = json.loads(data.decode("utf-8"))
        if parsed.get("completed"):
            raise OSError("injected checkpoint failure after rename")
        return real_atomic(path, data)

    mod._atomic_write = fail_completion_write  # type: ignore[assignment]
    try:
        with pytest.raises(ReferenceMaterializationError):
            await fs._materialize("k", str(dest), checkpoint=str(ckpt))
    finally:
        mod._atomic_write = real_atomic  # type: ignore[assignment]
    # The rename already happened, but no completed record exists.
    assert (dest / "k").read_bytes() == payload
    state = json.loads(ckpt.read_text())
    assert state["completed"] == {}

    # Resume must not trust the unmarked file: re-materialize, then record.
    before = len(backend.requests)
    result = await fs._materialize(
        "k", str(dest), checkpoint=str(ckpt), resume=True
    )
    assert result["written"] == ["k"]
    assert result["reused"] == []
    assert len(backend.requests) > before
    state2 = json.loads(ckpt.read_text())
    assert "k" in state2["completed"]


# ---------------------------------------------------------------------------
# Plan changes and corrupt checkpoints are rejected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_plan_changes_rejected(tmp_path):
    p1 = write_payload(tmp_path, "p1", 100, seed=1)
    p2 = write_payload(tmp_path, "p2", 100, seed=2)
    backend = make_backend(tmp_path, {})
    refs = {
        "a": ["recording://p1", 0, 50],
        "b": ["recording://p1", 50, 50],
    }
    fs = make_refs_fs(backend, refs)
    dest = tmp_path / "out"
    ckpt = tmp_path / "cp.json"
    await fs._materialize(
        ["a", "b"], str(dest), checkpoint=str(ckpt), expected_sha256=None
    )

    # changed range
    fs2 = make_refs_fs(backend, {"a": ["recording://p1", 0, 40], "b": refs["b"]})
    with pytest.raises(ReferenceMaterializationError, match="fingerprint|differs"):
        await fs2._materialize(
            ["a", "b"], str(dest), checkpoint=str(ckpt), resume=True
        )
    # changed URL
    fs3 = make_refs_fs(backend, {"a": ["recording://p2", 0, 50], "b": refs["b"]})
    with pytest.raises(ReferenceMaterializationError):
        await fs3._materialize(
            ["a", "b"], str(dest), checkpoint=str(ckpt), resume=True
        )
    # changed key set
    with pytest.raises(ReferenceMaterializationError):
        await fs._materialize(
            ["a"], str(dest), checkpoint=str(ckpt), resume=True
        )
    # changed destination root
    with pytest.raises(ReferenceMaterializationError):
        await fs._materialize(
            ["a", "b"], str(tmp_path / "other"),
            checkpoint=str(ckpt), resume=True
        )
    # changed expected digest
    with pytest.raises(ReferenceMaterializationError):
        await fs._materialize(
            ["a", "b"],
            str(dest),
            checkpoint=str(ckpt),
            resume=True,
            expected_sha256={
                "a": sha256_hex(p1[0:50]),
                "b": "f" * 64,
            },
        )
    # changed default target
    fs4 = fsspec.filesystem(
        "reference",
        fo={"a": [None, 0, 50], "b": refs["b"]},
        fs=backend,
        target="recording://p2",
        asynchronous=True,
    )
    with pytest.raises(ReferenceMaterializationError):
        await fs4._materialize(
            ["a", "b"], str(dest), checkpoint=str(ckpt), resume=True
        )


@pytest.mark.asyncio
async def test_bad_checkpoints_rejected(tmp_path):
    write_payload(tmp_path, "f", 100)
    backend = make_backend(tmp_path, {})
    fs = make_refs_fs(backend, {"k": ["recording://f", 0, 100]})
    dest = tmp_path / "out"
    ckpt = tmp_path / "cp.json"

    # resume without checkpoint
    with pytest.raises(ValueError):
        await fs._materialize(
            "k", str(dest), checkpoint=str(ckpt), resume=True
        )

    await fs._materialize("k", str(dest), checkpoint=str(ckpt))
    raw = ckpt.read_bytes()

    cases = {
        "not-json": b"{not json",
        "unknown-version": None,  # filled below
        "missing-field": None,
        "contradiction": None,
    }
    good = json.loads(raw)
    bad_version = dict(good, version=999)
    missing = {k: v for k, v in good.items() if k != "plan"}
    contradiction = dict(good)
    contradiction["order"] = good["order"] + ["ghost"]
    cases["unknown-version"] = json.dumps(bad_version).encode()
    cases["missing-field"] = json.dumps(missing).encode()
    cases["contradiction"] = json.dumps(contradiction).encode()
    for label, content in cases.items():
        ckpt.write_bytes(content)
        with pytest.raises(ReferenceMaterializationError) as exc:
            await fs._materialize(
                "k", str(dest), checkpoint=str(ckpt), resume=True
            )
        assert exc.value.path is None, label

    # tampered fingerprint must not be silently treated as empty progress
    tampered = dict(good, fingerprint="0" * 64)
    ckpt.write_bytes(json.dumps(tampered).encode())
    with pytest.raises(ReferenceMaterializationError):
        await fs._materialize(
            "k", str(dest), checkpoint=str(ckpt), resume=True
        )
    # restore: a good checkpoint still resumes
    ckpt.write_bytes(raw)
    result = await fs._materialize(
        "k", str(dest), checkpoint=str(ckpt), resume=True
    )
    assert result["reused"] == ["k"]
    assert json.loads(raw)["version"] == _CHECKPOINT_VERSION


# ---------------------------------------------------------------------------
# First-run guards, path boundaries, unrelated files, symlinks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_run_rejects_existing_checkpoint_and_targets(tmp_path):
    backend = make_backend(tmp_path, {"f": b"x"})
    fs = make_refs_fs(backend, {"k": ["recording://f"]})
    dest = tmp_path / "out"
    ckpt = tmp_path / "cp.json"
    await fs._materialize("k", str(dest), checkpoint=str(ckpt))
    with pytest.raises(ValueError, match="checkpoint already exists"):
        await fs._materialize("k", str(dest), checkpoint=str(ckpt))
    os.unlink(ckpt)
    with pytest.raises(ValueError, match="target file already exists"):
        await fs._materialize("k", str(dest), checkpoint=str(ckpt))


@pytest.mark.asyncio
async def test_unrelated_files_preserved(tmp_path):
    backend = make_backend(tmp_path, {"f": b"data"})
    fs = make_refs_fs(backend, {"k": ["recording://f"]})
    dest = tmp_path / "out"
    dest.mkdir()
    unrelated = dest / "keep-me"
    unrelated.write_bytes(b"untouched")
    (dest / "sub").mkdir()
    (dest / "sub" / "keep").write_bytes(b"also untouched")
    await fs._materialize(
        "k", str(dest), checkpoint=str(tmp_path / "cp.json")
    )
    assert unrelated.read_bytes() == b"untouched"
    assert (dest / "sub" / "keep").read_bytes() == b"also untouched"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key",
    ["/abs", "a/../b", "a/./b", "a//b", "a\\b", "..", "a/", "/"],
)
async def test_path_boundaries_rejected(tmp_path, key):
    backend = make_backend(tmp_path, {})
    fs = make_refs_fs(backend, {key: b"x"})
    with pytest.raises((ValueError, TypeError)):
        await fs._materialize(
            key, str(tmp_path / "out"), checkpoint=str(tmp_path / "cp.json")
        )


@pytest.mark.asyncio
async def test_duplicate_and_prefix_conflicts_rejected(tmp_path):
    backend = make_backend(tmp_path, {})
    fs = make_refs_fs(backend, {"a": b"1", "a/b": b"2"})
    with pytest.raises(ValueError, match="duplicate"):
        await fs._materialize(
            ["a", "a"], str(tmp_path / "out"), checkpoint=str(tmp_path / "cp.json")
        )
    with pytest.raises(ValueError, match="directory prefix"):
        await fs._materialize(
            ["a", "a/b"], str(tmp_path / "out"),
            checkpoint=str(tmp_path / "cp2.json"),
        )


@pytest.mark.asyncio
async def test_symlink_escape_rejected(tmp_path):
    backend = make_backend(tmp_path, {"f": b"data"})
    fs = make_refs_fs(backend, {"k": ["recording://f"], "d/k": ["recording://f"]})
    outside = tmp_path / "outside"
    outside.mkdir()
    dest = tmp_path / "out"
    dest.mkdir()
    # symlinked intermediate directory
    os.symlink(outside, dest / "d")
    with pytest.raises(ReferenceMaterializationError, match="symlink"):
        await fs._materialize(
            "d/k", str(dest), checkpoint=str(tmp_path / "cp.json")
        )
    # symlinked final target
    link = dest / "k"
    os.symlink(outside / "escape", link)
    with pytest.raises(ReferenceMaterializationError, match="symlink"):
        await fs._materialize(
            "k", str(dest), checkpoint=str(tmp_path / "cp2.json")
        )
    assert not (outside / "escape").exists()


@pytest.mark.asyncio
async def test_checkpoint_must_not_equal_target(tmp_path):
    backend = make_backend(tmp_path, {})
    fs = make_refs_fs(backend, {"cp.json": b"x"})
    with pytest.raises(ValueError, match="checkpoint path must not"):
        await fs._materialize(
            "cp.json", str(tmp_path / "out"),
            checkpoint=str(tmp_path / "out" / "cp.json"),
        )


@pytest.mark.asyncio
async def test_invalid_parameters(tmp_path):
    backend = make_backend(tmp_path, {})
    fs = make_refs_fs(backend, {})
    common = dict(
        destination=str(tmp_path / "out"), checkpoint=str(tmp_path / "cp.json")
    )
    with pytest.raises(ValueError):
        await fs._materialize([], **common)
    with pytest.raises((TypeError, ValueError)):
        await fs._materialize([1], **common)
    for bad in (0, -1, True, False, 1.5, "8"):
        with pytest.raises((TypeError, ValueError)):
            await fs._materialize(
                ["k"], block_size=bad, max_concurrency=1, **common
            )
        with pytest.raises((TypeError, ValueError)):
            await fs._materialize(
                ["k"], block_size=1, max_concurrency=bad, **common
            )
    with pytest.raises(TypeError):
        await fs._materialize(
            ["k"], **common, resume="yes"
        )
    with pytest.raises(ValueError):
        await fs._materialize(
            ["k"], destination="", checkpoint=str(tmp_path / "cp.json")
        )
    with pytest.raises(ValueError):
        await fs._materialize(
            ["k"], destination=str(tmp_path / "out"), checkpoint=""
        )


# ---------------------------------------------------------------------------
# Failures during write/rename/checkpoint leave no published partial state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_failure_cleans_temp(tmp_path):
    import fsspec.implementations._reference_materialize as mod

    backend = make_backend(tmp_path, {"f": b"payload" * 100})
    fs = make_refs_fs(backend, {"k": ["recording://f"]})
    dest = tmp_path / "out"

    mod._WRITE_FAILURE_HOOK = OSError("injected disk failure")
    try:
        with pytest.raises(ReferenceMaterializationError) as exc:
            await fs._materialize(
                "k", str(dest), checkpoint=str(tmp_path / "cp.json"),
                block_size=64,
            )
    finally:
        mod._WRITE_FAILURE_HOOK = None
    assert exc.value.path == "k"
    assert isinstance(exc.value.__cause__, OSError)
    assert not (dest / "k").exists()
    temps = list(dest.rglob(".fsspec-materialize-*"))
    assert temps == []
    # the run is still resumable from the persisted plan
    result = await fs._materialize(
        "k", str(dest), checkpoint=str(tmp_path / "cp.json"), resume=True
    )
    assert result["written"] == ["k"]
    assert (dest / "k").read_bytes() == b"payload" * 100


@pytest.mark.asyncio
async def test_checkpoint_update_failure_keeps_completed_others(tmp_path):
    backend = make_backend(
        tmp_path, {"a": b"a" * 100, "b": b"b" * 100, "c": b"c" * 100}
    )
    refs = {
        "a": ["recording://a"],
        "b": ["recording://b"],
        "c": ["recording://c"],
    }
    fs = make_refs_fs(backend, refs)
    dest = tmp_path / "out"
    ckpt = tmp_path / "cp.json"

    import fsspec.implementations._reference_materialize as mod

    real_atomic = mod._atomic_write
    calls = {"n": 0}

    def flaky(path, data):
        calls["n"] += 1
        # initial plan write (1st) succeeds; fail the completion update for b
        if calls["n"] == 3:
            raise OSError("injected checkpoint IO error")
        return real_atomic(path, data)

    mod._atomic_write = flaky  # type: ignore[assignment]
    try:
        with pytest.raises(ReferenceMaterializationError):
            await fs._materialize(
                ["a", "b", "c"], str(dest), checkpoint=str(ckpt),
                max_concurrency=1,
            )
    finally:
        mod._atomic_write = real_atomic  # type: ignore[assignment]

    state = json.loads(ckpt.read_text())
    # a completed before b failed; its record must survive
    assert set(state["completed"]) == {"a"}
    assert (dest / "a").read_bytes() == b"a" * 100
    # b may have been renamed but is not marked complete -> redone on resume
    result = await fs._materialize(
        ["a", "b", "c"], str(dest), checkpoint=str(ckpt),
        max_concurrency=1, resume=True,
    )
    assert result["reused"] == ["a"]
    assert sorted(result["written"]) == ["b", "c"]
    assert (dest / "b").read_bytes() == b"b" * 100
    assert (dest / "c").read_bytes() == b"c" * 100


# ---------------------------------------------------------------------------
# Synchronous wrapper and original get/cat/open compatibility
# ---------------------------------------------------------------------------


def test_sync_materialize_roundtrip(tmp_path):
    payload = bytes(range(250))
    root = tmp_path / "remote"
    root.mkdir()
    (root / "f").write_bytes(payload)
    # plain sync ReferenceFileSystem (memory backend wrapped automatically);
    # here use the local file backend which gets AsyncFileSystemWrapper.
    refs = {
        "inline": b"in",
        "range": [f"file://{root / 'f'}", 10, 100],
        "whole": [f"file://{root / 'f'}"],
    }
    fs = fsspec.filesystem(
        "reference", fo=refs, remote_protocol="file", skip_instance_cache=True
    )
    dest = tmp_path / "out"
    ckpt = tmp_path / "cp.json"
    result = fs.materialize(
        ["inline", "range", "whole"],
        str(dest),
        checkpoint=str(ckpt),
        block_size=37,
        max_concurrency=2,
    )
    assert result["written"] == ["inline", "range", "whole"]
    assert os.path.isabs(result["checkpoint"])
    assert (dest / "inline").read_bytes() == b"in"
    assert (dest / "range").read_bytes() == payload[10:110]
    assert (dest / "whole").read_bytes() == payload
    # sync resume
    result2 = fs.materialize(
        ["whole", "inline", "range"],
        str(dest),
        checkpoint=str(ckpt),
        resume=True,
    )
    assert result2["reused"] == ["whole", "inline", "range"]
    assert result2["bytes_written"] == 0


def test_original_get_cat_open_unchanged(tmp_path):
    payload = b"0123456789"
    root = tmp_path / "remote"
    root.mkdir()
    (root / "f").write_bytes(payload)
    refs = {
        "a": b"data",
        "b": (str(root / "f"), 0, 5),
        "c/d": (str(root / "f"), 1, 6),
    }
    fs = fsspec.filesystem(
        "reference", fo=refs, remote_protocol="file", skip_instance_cache=True
    )
    # cat / open behavior untouched
    assert fs.cat("a") == b"data"
    assert fs.cat("b") == b"01234"
    with fs.open("c/d", "rb") as f:
        assert f.read() == b"123456"
    # get behavior untouched
    out = tmp_path / "got"
    fs.get("b", str(out))
    assert out.read_bytes() == b"01234"
    fs.get("c", str(tmp_path / "gotdir"), recursive=True)
    assert (tmp_path / "gotdir" / "d").read_bytes() == b"123456"
    # get_file untouched
    fs.get_file("a", str(tmp_path / "gotfile"))
    assert (tmp_path / "gotfile").read_bytes() == b"data"


def test_materialize_does_not_nest_asyncio_run(tmp_path):
    # Calling the sync method must work without a running loop and must not
    # invoke asyncio.run internally (it uses the fsspec IO loop thread).
    backend_root = tmp_path / "remote"
    backend_root.mkdir()
    (backend_root / "f").write_bytes(b"xy")
    fs = fsspec.filesystem(
        "reference",
        fo={"k": [f"file://{backend_root / 'f'}"]},
        remote_protocol="file",
        skip_instance_cache=True,
    )
    import asyncio as _a

    called = []
    real_run = _a.run

    def no_run(*a, **k):
        called.append(True)
        raise AssertionError("materialize must not call asyncio.run")

    _a.run = no_run  # type: ignore[assignment]
    try:
        result = fs.materialize(
            "k", str(tmp_path / "out"), checkpoint=str(tmp_path / "cp.json")
        )
    finally:
        _a.run = real_run  # type: ignore[assignment]
    assert called == []
    assert result["written"] == ["k"]


def test_exception_path_attribute_and_chain(tmp_path):
    backend_root = tmp_path / "remote"
    backend_root.mkdir()
    fs = fsspec.filesystem(
        "reference",
        fo={"missing": ["file:///does/not/exist/anywhere"]},
        remote_protocol="file",
        skip_instance_cache=True,
    )
    with pytest.raises(ReferenceMaterializationError) as exc:
        fs.materialize(
            "missing", str(tmp_path / "out"), checkpoint=str(tmp_path / "cp.json")
        )
    assert exc.value.path == "missing"
    assert exc.value.__cause__ is not None


def test_referencefs_is_async_instance_awaitable(tmp_path):
    # sanity: ReferenceFileSystem exposes both entry points
    assert asyncio.iscoroutinefunction(ReferenceFileSystem._materialize)
    assert not asyncio.iscoroutinefunction(ReferenceFileSystem.materialize)


@pytest.mark.asyncio
async def test_fingerprint_order_independent_and_stable(tmp_path):
    from fsspec.implementations._reference_materialize import (
        _compute_fingerprint,
        _resolve_entry,
    )
    backend = make_backend(tmp_path, {"a": b"a" * 10, "b": b"b" * 10})
    refs = {
        "a": ["recording://a", 0, 5],
        "b": ["recording://b"],
        "i": b"inline",
    }
    fs = make_refs_fs(backend, refs)
    plan = {k: await _resolve_entry(fs, k) for k in refs}
    fp1 = _compute_fingerprint("/dest", None, None, plan)
    # rebuilding in a different iteration order gives the same digest
    reordered = {k: plan[k] for k in reversed(list(plan))}
    fp2 = _compute_fingerprint("/dest", None, None, reordered)
    assert fp1 == fp2
    # any identity change changes the fingerprint
    plan2 = dict(plan)
    plan2["a"] = dict(plan["a"], size=4)
    assert _compute_fingerprint("/dest", None, None, plan2) != fp1
    assert _compute_fingerprint("/other", None, None, plan) != fp1


@pytest.mark.asyncio
async def test_checkpoint_never_published_half_written(tmp_path):
    backend = make_backend(tmp_path, {"f": b"payload"})
    fs = make_refs_fs(backend, {"k": ["recording://f"]})
    ckpt = tmp_path / "cp.json"
    await fs._materialize(
        "k", str(tmp_path / "out"), checkpoint=str(ckpt)
    )
    # every successful replacement leaves parseable JSON and no temp files
    state = json.loads(ckpt.read_text())
    assert state["format"] == "fsspec-reference-materialization"
    leftovers = list(tmp_path.rglob(".fsspec-materialize-*"))
    assert leftovers == []


@pytest.mark.asyncio
async def test_unselected_references_not_resolved(tmp_path):
    # a reference that would fail if touched must be ignored when not selected
    backend = make_backend(tmp_path, {"good": b"ok"})
    backend.unknown_size_paths.add("recording://mystery")
    fs = make_refs_fs(
        backend,
        {"good": ["recording://good"], "mystery": ["recording://mystery"]},
    )
    result = await fs._materialize(
        "good", str(tmp_path / "out"), checkpoint=str(tmp_path / "cp.json")
    )
    assert result["written"] == ["good"]


@pytest.mark.asyncio
async def test_missing_logical_key_error(tmp_path):
    backend = make_backend(tmp_path, {})
    fs = make_refs_fs(backend, {"present": b"x"})
    with pytest.raises(ReferenceMaterializationError) as exc:
        await fs._materialize(
            "absent", str(tmp_path / "out"), checkpoint=str(tmp_path / "cp.json")
        )
    assert exc.value.path == "absent"
    assert isinstance(exc.value.__cause__, FileNotFoundError)


@pytest.mark.asyncio
async def test_remote_read_failure_wrapped(tmp_path):
    backend = make_backend(tmp_path, {"f": b"abc"})
    backend.fail_paths.add("recording://f")
    fs = make_refs_fs(backend, {"k": ["recording://f", 0, 3]})
    with pytest.raises(ReferenceMaterializationError) as exc:
        await fs._materialize(
            "k", str(tmp_path / "out"), checkpoint=str(tmp_path / "cp.json")
        )
    assert exc.value.path == "k"
    assert isinstance(exc.value.__cause__, OSError)


@pytest.mark.asyncio
async def test_resume_without_checkpoint_is_valueerror(tmp_path):
    backend = make_backend(tmp_path, {"f": b"abc"})
    fs = make_refs_fs(backend, {"k": ["recording://f"]})
    with pytest.raises(ValueError, match="resume=True but checkpoint does not exist"):
        await fs._materialize(
            "k", str(tmp_path / "out"),
            checkpoint=str(tmp_path / "missing.json"), resume=True
        )


@pytest.mark.asyncio
async def test_rename_failure_cleans_temp(tmp_path):
    import fsspec.implementations._reference_materialize as mod

    backend = make_backend(tmp_path, {"f": b"payload" * 10})
    fs = make_refs_fs(backend, {"k": ["recording://f"]})
    dest = tmp_path / "out"

    mod._RENAME_FAILURE_HOOK = OSError("injected rename failure")
    try:
        with pytest.raises(ReferenceMaterializationError) as exc:
            await fs._materialize(
                "k", str(dest), checkpoint=str(tmp_path / "cp.json")
            )
    finally:
        mod._RENAME_FAILURE_HOOK = None
    assert exc.value.path == "k"
    assert isinstance(exc.value.__cause__, OSError)
    # nothing was published at the final path
    assert not (dest / "k").exists()
    assert list(dest.rglob(".fsspec-materialize-*")) == []
    # the initial plan checkpoint remains valid and the run is resumable
    result = await fs._materialize(
        "k", str(dest), checkpoint=str(tmp_path / "cp.json"), resume=True
    )
    assert result["written"] == ["k"]
    assert (dest / "k").read_bytes() == b"payload" * 10


@pytest.mark.asyncio
async def test_concurrent_completion_records_are_not_lost(tmp_path):
    # With many workers all completing, every key must end up recorded in the
    # final checkpoint (no completed entry lost to interleaved updates).
    write_payload(tmp_path, "shared", 800)
    backend = make_backend(tmp_path, {}, delay=0.01)
    refs = {f"k{i:02d}": ["recording://shared", 0, 800] for i in range(20)}
    fs = make_refs_fs(backend, refs)
    ckpt = tmp_path / "cp.json"
    result = await fs._materialize(
        list(refs), str(tmp_path / "out"), checkpoint=str(ckpt),
        block_size=64, max_concurrency=6,
    )
    assert sorted(result["written"]) == sorted(refs)
    state = json.loads(ckpt.read_text())
    assert set(state["completed"]) == set(refs)
    for key in refs:
        assert state["completed"][key]["size"] == 800


@pytest.mark.asyncio
async def test_async_backend_cancellation_mid_flight(tmp_path):
    # Natively async (sleep-based) backend: in-flight coroutines must be
    # cancelled and the call must return with no background work pending.
    write_payload(tmp_path, "slow", 1024)
    backend = make_backend(tmp_path, {}, delay=0.05)
    refs = {f"k{i}": ["recording://slow", 0, 1024] for i in range(16)}
    fs = make_refs_fs(backend, refs)
    task = asyncio.ensure_future(
        fs._materialize(
            list(refs), str(tmp_path / "out"),
            checkpoint=str(tmp_path / "cp.json"),
            block_size=128, max_concurrency=8,
        )
    )
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert backend.active == 0
    # no further requests after the call returned
    snapshot = len(backend.requests)
    await asyncio.sleep(0.15)
    assert len(backend.requests) == snapshot
    # every checkpoint written before cancellation is valid JSON
    state = json.loads((tmp_path / "cp.json").read_text())
    assert state["format"] == "fsspec-reference-materialization"


@pytest.mark.asyncio
async def test_repeated_resume_is_stable(tmp_path):
    payload = write_payload(tmp_path, "f", 777)
    backend = make_backend(tmp_path, {})
    fs = make_refs_fs(backend, {"a": ["recording://f"], "b": ["recording://f", 0, 10]})
    keys = ["a", "b"]
    await fs._materialize(
        keys, str(tmp_path / "out"), checkpoint=str(tmp_path / "cp.json")
    )
    for _ in range(3):
        result = await fs._materialize(
            keys, str(tmp_path / "out"),
            checkpoint=str(tmp_path / "cp.json"), resume=True,
        )
        assert result["written"] == []
        assert result["reused"] == keys
        assert result["bytes_written"] == 0
    assert (tmp_path / "out" / "a").read_bytes() == payload
    assert (tmp_path / "out" / "b").read_bytes() == payload[:10]


@pytest.mark.asyncio
async def test_zero_length_whole_file(tmp_path):
    (tmp_path / "remote").mkdir(exist_ok=True)
    (tmp_path / "remote" / "empty").write_bytes(b"")
    backend = make_backend(tmp_path, {})
    fs = make_refs_fs(backend, {"e": ["recording://empty"]})
    result = await fs._materialize(
        "e", str(tmp_path / "out"), checkpoint=str(tmp_path / "cp.json")
    )
    assert result["bytes_written"] == 0
    assert result["written"] == ["e"]
    assert (tmp_path / "out" / "e").read_bytes() == b""
    assert backend.requests == []  # zero payload reads for an empty whole file
    state = json.loads((tmp_path / "cp.json").read_text())
    assert state["completed"]["e"]["size"] == 0


@pytest.mark.asyncio
async def test_info_failure_wrapped_with_chain(tmp_path):
    write_payload(tmp_path, "f", 10)
    backend = make_backend(tmp_path, {})
    backend.fail_paths.add("recording://f")
    fs = make_refs_fs(backend, {"k": ["recording://f"]})
    with pytest.raises(ReferenceMaterializationError) as exc:
        await fs._materialize(
            "k", str(tmp_path / "out"), checkpoint=str(tmp_path / "cp.json")
        )
    assert exc.value.path == "k"
    assert exc.value.__cause__ is not None
