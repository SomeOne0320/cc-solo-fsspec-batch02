"""Resumable, bounded-concurrency materialization for ReferenceFileSystem.

This module implements :meth:`ReferenceFileSystem.materialize`, an interface
that materializes selected *logical* reference keys into ordinary local files
without ever collecting whole logical files (or the whole selection) into
memory.  It is deliberately separate from ``get``/``cat``/``open``: none of
those existing code paths or their default behavior are changed.

The implementation only relies on:

* the reference parsing semantics already exposed by
  :meth:`ReferenceFileSystem._cat_common` (V0/V1 references, rendered
  templates, default ``target``, plain/base64 inline values, whole-file and
  byte-range references), and
* the async ``_cat_file`` / ``_info`` methods of the per-protocol backends in
  ``ReferenceFileSystem.fss`` (synchronous backends are wrapped by
  :class:`~fsspec.implementations.asyn_wrapper.AsyncFileSystemWrapper`, i.e.
  their blocking work runs in worker threads).

See ``docs/source/reference_materialize.rst`` for the full progress/checkpoint
contract and the documented limitations.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import hashlib
import json
import logging
import os
import tempfile
from typing import Any

from fsspec.asyn import sync
from fsspec.core import split_protocol

logger = logging.getLogger("fsspec.reference.materialize")


class _TrackingExecutor(concurrent.futures.ThreadPoolExecutor):
    """ThreadPoolExecutor that retains every submitted work future.

    Retention is O(number of blocking calls in one run) and futures are
    released as soon as they finish, so the set only ever holds in-flight
    work.  Tracking at this level survives asyncio cancelling its own wrapper
    futures (which happens immediately on task cancellation even though the
    worker thread keeps running).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracked: set[concurrent.futures.Future] = set()

    def submit(self, fn, /, *args, **kwargs):  # type: ignore[override]
        future = super().submit(fn, *args, **kwargs)
        self.tracked.add(future)
        future.add_done_callback(self.tracked.discard)
        return future


class _BlockingWork:
    """Registry of blocking work issued by one materialize() call.

    Synchronous backends are wrapped with
    :class:`~fsspec.implementations.asyn_wrapper.AsyncFileSystemWrapper`, whose
    ``_cat_file`` runs the real blocking call inside ``asyncio.to_thread``.
    When the owning asyncio task is cancelled, asyncio resolves the wrapper
    future immediately even though the worker thread is still running.  To be
    able to wait for that work, each materialize() call installs its own
    :class:`_TrackingExecutor` as the loop's default executor for the duration
    of the call (asyncio.to_thread / run_in_executor with ``None`` target it),
    and joins the outstanding futures during shutdown.  The loop's previous
    default executor (usually lazily created and shared) is restored
    afterwards.
    """

    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.executor = _TrackingExecutor(
            thread_name_prefix="fsspec-materialize"
        )
        self._previous_executor = None

    def install(self) -> None:
        self._previous_executor = getattr(
            self.loop, "_default_executor", None
        )
        self.loop._default_executor = self.executor

    def restore(self) -> None:
        try:
            self.loop._default_executor = self._previous_executor
        finally:
            # All tracked work has been awaited by the time restore runs;
            # do not cancel futures, just release the pool's idle threads.
            self.executor.shutdown(wait=False, cancel_futures=False)

    def submit(self, func, *args, **kwargs):
        return self.loop.run_in_executor(
            self.executor, lambda: func(*args, **kwargs)
        )

    async def wait_all(self) -> None:
        """Block (shielded) until every issued blocking call has finished."""
        pending = [f for f in self.executor.tracked if not f.done()]
        if not pending:
            return

        def _join():
            concurrent.futures.wait(pending)

        # Join from yet another thread so the event loop stays responsive to
        # the cancellation signal while we shield the wait itself.
        await asyncio.shield(self.loop.run_in_executor(None, _join))


async def _to_thread(blocking: "_BlockingWork", func, *args, **kwargs):
    """Run blocking ``func`` on the run-scoped executor and await the result."""
    return await blocking.submit(func, *args, **kwargs)


_CHECKPOINT_FORMAT = "fsspec-reference-materialization"
_CHECKPOINT_VERSION = 1
_TMP_PREFIX = ".fsspec-materialize-"

# Test hooks: when set to an exception instance, the next payload write /
# final rename of a materialize run raises it instead of performing the
# operation.  Production code leaves these as None.
_WRITE_FAILURE_HOOK: Exception | None = None
_RENAME_FAILURE_HOOK: Exception | None = None

# Chunk size used when *verifying* (stream-reading) already completed local
# files on resume; independent of the network block_size parameter.
_VERIFY_BLOCK_SIZE = 1024 * 1024


def _mkstemp_tracked(
    file_registry: set, fd_registry: set, parent: str
) -> tuple[int, str]:
    """Create a temp file and register it *before* returning to the loop.

    Registering both the path and the open descriptor from inside the worker
    thread closes the cancellation window in which ``mkstemp`` has created
    the file but the awaiting coroutine has not yet stored its results: even
    if the await is cancelled, shutdown cleanup sees the path (and can close
    the descriptor) and removes it.
    """
    fd, path = tempfile.mkstemp(prefix=_TMP_PREFIX, dir=parent)
    file_registry.add(path)
    fd_registry.add(fd)
    return fd, path


class ReferenceMaterializationError(Exception):
    """Raised when materializing a single logical key fails.

    The :attr:`path` attribute equals the offending logical key.  It is
    ``None`` for failures that are not tied to a particular key (global
    parameter, plan or checkpoint validation failures).
    """

    def __init__(self, path, message, *args):
        if path is None:
            text = message
        else:
            text = f"{message} (logical key: {path!r})"
        super().__init__(text, *args)
        self.path = path


def _is_positive_int(value) -> bool:
    # bool is a subclass of int and must be rejected explicitly.
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _canonical_json(obj: Any) -> str:
    """Stable JSON text used for the plan fingerprint."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Logical-key / local-path validation
# ---------------------------------------------------------------------------


def _validate_logical_key(key) -> str:
    if not isinstance(key, str):
        raise TypeError(
            "paths must be a logical-key string or a sequence of logical-key "
            f"strings, got element of type {type(key).__name__}"
        )
    if key == "":
        raise ValueError("logical keys must be non-empty")
    if "\\" in key:
        raise ValueError(
            f"logical key {key!r} must not contain backslashes"
        )
    if key.startswith("/"):
        raise ValueError(f"logical key {key!r} must be relative, not absolute")
    segments = key.split("/")
    if any(seg == "" for seg in segments):
        raise ValueError(f"logical key {key!r} contains an empty path segment")
    if any(seg in ("..", ".") for seg in segments):
        raise ValueError(
            f"logical key {key!r} must not contain '.' or '..' segments"
        )
    return key


def _check_key_conflicts(keys: list[str]) -> None:
    """Reject duplicates and file/directory prefix conflicts up front."""
    seen = set()
    for key in keys:
        if key in seen:
            raise ValueError(f"duplicate logical key in paths: {key!r}")
        seen.add(key)

    prefixes = set()
    for key in keys:
        segments = key.split("/")
        for i in range(1, len(segments)):
            prefixes.add("/".join(segments[:i]))
    for key in keys:
        if key in prefixes:
            raise ValueError(
                f"logical key {key!r} is both a file and a directory prefix "
                "of another selected key"
            )


def _check_no_symlink_escape(root: str, rel_segments: list[str], where: str) -> None:
    """Ensure existing path components under root are not symlinks.

    Only already-existing components need inspecting: components that do not
    exist yet will be created as real directories.  After creation/rename the
    final location is checked again with ``realpath`` to defend against
    symlinked destination roots.
    """
    current = root
    for segment in rel_segments:
        current = os.path.join(current, segment)
        if os.path.islink(current):
            raise ReferenceMaterializationError(
                None,
                f"{where} path {current!r} passes through an existing symlink, "
                "which would allow escaping the destination root",
            )


def _safe_local_path(root: str, key: str) -> str:
    candidate = os.path.abspath(os.path.join(root, *key.split("/")))
    # root is itself abspath-normalized, so a common prefix implies containment
    if os.path.commonpath([candidate, root]) != root:
        raise ValueError(
            f"logical key {key!r} maps outside the destination root"
        )
    return candidate


# ---------------------------------------------------------------------------
# Reference resolution -> serializable plan entries
# ---------------------------------------------------------------------------


def _decode_inline(part) -> bytes:
    """Mirror the inline handling in ReferenceFileSystem._cat_common."""
    if isinstance(part, str):
        part = part.encode()
    if hasattr(part, "to_bytes"):
        part = part.to_bytes()
    if isinstance(part, bytes) and part.startswith(b"base64:"):
        part = base64.b64decode(part[7:])
    return part


async def _resolve_entry(fs, key: str) -> dict:
    """Resolve one logical key into a serializable plan entry.

    Reuses :meth:`ReferenceFileSystem._cat_common` so every accepted reference
    shape (V0/V1, rendered templates, default target, inline, whole-file,
    range, zero-length) follows the exact same parsing semantics as
    ``cat``/``open``.
    """
    try:
        part_or_url, start0, end0 = fs._cat_common(key)
    except FileNotFoundError as exc:
        raise ReferenceMaterializationError(
            key, "logical key is not present in the references"
        ) from exc
    except ReferenceMaterializationError:
        raise
    except Exception as exc:
        raise ReferenceMaterializationError(
            key, "failed to parse reference"
        ) from exc

    if isinstance(part_or_url, bytes):
        payload = _decode_inline(part_or_url)
        return {
            "kind": "inline",
            "inline_b64": base64.b64encode(payload).decode("ascii"),
            "size": len(payload),
        }

    url = part_or_url
    protocol, _ = split_protocol(url)
    backend = fs.fss.get(protocol)
    if backend is None:
        raise ReferenceMaterializationError(
            key, f"no backend available for protocol {protocol!r} in {url!r}"
        )

    uses_default_target = url == fs.target
    if start0 is None and end0 is None:
        # Whole-file reference: the size must be known from the backend.
        try:
            info = await backend._info(url)
        except Exception as exc:
            raise ReferenceMaterializationError(
                key, f"backend info() failed for whole-file target {url!r}"
            ) from exc
        size = info.get("size") if isinstance(info, dict) else None
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ReferenceMaterializationError(
                key,
                f"whole-file target {url!r} has no known non-negative integer "
                f"size (got {size!r}); materializing an unknown-size whole file "
                "is not supported",
            )
        return {
            "kind": "url",
            "url": url,
            "offset": None,
            "size": size,
            "default_target": uses_default_target,
            "protocol": protocol,
        }

    # Range reference [url, offset, size] (zero-length ranges included).
    offset = int(start0)
    size = int(end0 - start0)
    if offset < 0 or size < 0:
        raise ReferenceMaterializationError(
            key, f"invalid resolved range offset={offset} size={size}"
        )
    return {
        "kind": "url",
        "url": url,
        "offset": offset,
        "size": size,
        "default_target": uses_default_target,
        "protocol": protocol,
    }


def _entry_identity(entry: dict) -> dict:
    """Fingerprint-relevant content of a plan entry (protocol is local info)."""
    if entry["kind"] == "inline":
        return {"kind": "inline", "inline_b64": entry["inline_b64"]}
    return {
        "kind": "url",
        "url": entry["url"],
        "offset": entry["offset"],
        "size": entry["size"],
        "default_target": entry["default_target"],
    }


def _compute_fingerprint(
    destination: str, target, expected: dict | None, plan: dict
) -> str:
    material = {
        "destination": destination,
        "target": target,
        "expected_sha256": expected,
        "entries": {key: _entry_identity(plan[key]) for key in sorted(plan)},
    }
    digest = hashlib.sha256(_canonical_json(material).encode("utf-8"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Atomic local file / checkpoint helpers
# ---------------------------------------------------------------------------


def _atomic_write(path: str, data: bytes) -> None:
    """Write bytes via a same-directory temp file and atomic replace."""
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=_TMP_PREFIX, dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _load_checkpoint(checkpoint_path: str) -> dict:
    try:
        with open(checkpoint_path, "rb") as f:
            raw = f.read()
    except FileNotFoundError as exc:
        raise ValueError(
            f"resume=True but checkpoint does not exist: {checkpoint_path!r}"
        ) from exc
    except OSError as exc:
        raise ReferenceMaterializationError(
            None, f"cannot read checkpoint {checkpoint_path!r}: {exc}"
        ) from exc
    try:
        state = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ReferenceMaterializationError(
            None, f"checkpoint {checkpoint_path!r} is not valid JSON"
        ) from exc
    if not isinstance(state, dict):
        raise ReferenceMaterializationError(
            None, "checkpoint root must be a JSON object"
        )
    return state


def _validate_checkpoint_structure(state: dict) -> None:
    required = {
        "format",
        "version",
        "fingerprint",
        "destination",
        "target",
        "expected_sha256",
        "plan",
        "completed",
        "order",
    }
    missing = required - state.keys()
    if missing:
        raise ReferenceMaterializationError(
            None,
            "checkpoint is incomplete, missing fields: "
            + ", ".join(sorted(missing)),
        )
    if state["format"] != _CHECKPOINT_FORMAT:
        raise ReferenceMaterializationError(
            None, f"unknown checkpoint format: {state['format']!r}"
        )
    if state["version"] != _CHECKPOINT_VERSION:
        raise ReferenceMaterializationError(
            None,
            f"unsupported checkpoint version: {state['version']!r}; "
            f"supported version is {_CHECKPOINT_VERSION}",
        )
    if not isinstance(state["fingerprint"], str):
        raise ReferenceMaterializationError(None, "checkpoint fingerprint is invalid")
    if not isinstance(state["destination"], str) or not state["destination"]:
        raise ReferenceMaterializationError(None, "checkpoint destination is invalid")
    if not isinstance(state["plan"], dict):
        raise ReferenceMaterializationError(None, "checkpoint plan is invalid")
    if not isinstance(state["completed"], dict):
        raise ReferenceMaterializationError(None, "checkpoint completed map is invalid")
    if not isinstance(state["order"], list) or any(
        not isinstance(k, str) for k in state["order"]
    ):
        raise ReferenceMaterializationError(None, "checkpoint order list is invalid")
    if set(state["order"]) != set(state["plan"]):
        raise ReferenceMaterializationError(
            None, "checkpoint order and plan keys are inconsistent"
        )
    for key, entry in state["plan"].items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            raise ReferenceMaterializationError(None, "checkpoint plan entry invalid")
        if entry.get("kind") == "inline":
            if not isinstance(entry.get("inline_b64"), str) or not isinstance(
                entry.get("size"), int
            ) or isinstance(entry.get("size"), bool) or entry["size"] < 0:
                raise ReferenceMaterializationError(
                    None, f"checkpoint inline entry for {key!r} is invalid"
                )
        elif entry.get("kind") == "url":
            if (
                not isinstance(entry.get("url"), str)
                or not isinstance(entry.get("size"), int)
                or isinstance(entry.get("size"), bool)
                or entry["size"] < 0
                or (
                    entry.get("offset") is not None
                    and (
                        not isinstance(entry.get("offset"), int)
                        or isinstance(entry.get("offset"), bool)
                        or entry["offset"] < 0
                    )
                )
                or not isinstance(entry.get("default_target"), bool)
            ):
                raise ReferenceMaterializationError(
                    None, f"checkpoint url entry for {key!r} is invalid"
                )
        else:
            raise ReferenceMaterializationError(
                None, f"checkpoint entry for {key!r} has unknown kind"
            )
    for key, rec in state["completed"].items():
        if key not in state["plan"] or not isinstance(rec, dict):
            raise ReferenceMaterializationError(
                None, f"checkpoint completed record for {key!r} is invalid"
            )
        if (
            not isinstance(rec.get("size"), int)
            or isinstance(rec.get("size"), bool)
            or rec["size"] < 0
            or not isinstance(rec.get("sha256"), str)
            or len(rec["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in rec["sha256"])
        ):
            raise ReferenceMaterializationError(
                None, f"checkpoint completed record for {key!r} is invalid"
            )
    expected = state["expected_sha256"]
    if expected is not None:
        if not isinstance(expected, dict) or any(
            k not in state["plan"]
            or not isinstance(v, str)
            or len(v) != 64
            or any(c not in "0123456789abcdef" for c in v)
            for k, v in expected.items()
        ):
            raise ReferenceMaterializationError(
                None, "checkpoint expected_sha256 map is invalid"
            )


async def _stream_file_sha256(
    path: str, blocking: "_BlockingWork"
) -> tuple[int, str]:
    def _read() -> tuple[int, "hashlib._Hash"]:
        h = hashlib.sha256()
        total = 0
        with open(path, "rb") as f:
            while True:
                chunk = f.read(_VERIFY_BLOCK_SIZE)
                if not chunk:
                    break
                h.update(chunk)
                total += len(chunk)
        return total, h

    total, h = await _to_thread(blocking, _read)
    return total, h.hexdigest()


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


async def materialize(
    fs,
    paths,
    destination,
    *,
    checkpoint,
    expected_sha256=None,
    block_size=1048576,
    max_concurrency=4,
    resume=False,
):
    """Async implementation backing :meth:`ReferenceFileSystem.materialize`."""
    # -- parameter validation (global errors, path is None) -----------------
    if not _is_positive_int(block_size):
        raise ValueError(
            "block_size must be a positive integer (booleans are not allowed), "
            f"got {block_size!r}"
        )
    if not _is_positive_int(max_concurrency):
        raise ValueError(
            "max_concurrency must be a positive integer (booleans are not "
            f"allowed), got {max_concurrency!r}"
        )
    if not isinstance(resume, bool):
        raise TypeError(f"resume must be a bool, got {type(resume).__name__}")
    if isinstance(paths, str):
        keys = [paths]
    else:
        try:
            keys = list(paths)
        except TypeError as exc:
            raise TypeError(
                "paths must be a logical-key string or a sequence of "
                "logical-key strings"
            ) from exc
    if len(keys) == 0:
        raise ValueError("paths must contain at least one logical key")
    for key in keys:
        _validate_logical_key(key)
    _check_key_conflicts(keys)

    if not isinstance(destination, str) or not destination:
        raise ValueError("destination must be a non-empty local path string")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("checkpoint must be a non-empty local path string")

    destination_abs = os.path.abspath(destination)
    checkpoint_abs = os.path.abspath(checkpoint)

    expected_norm = None
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, dict):
            raise TypeError(
                "expected_sha256 must be None or a mapping of logical key to "
                "64-character hex SHA256 digest"
            )
        missing = [k for k in keys if k not in expected_sha256]
        extra = [k for k in expected_sha256 if k not in keys]
        if missing or extra:
            raise ValueError(
                "expected_sha256 must cover exactly the selected logical keys; "
                f"missing={missing!r}, unexpected={extra!r}"
            )
        expected_norm = {}
        for key in keys:
            digest = expected_sha256[key]
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(
                    f"expected SHA256 for {key!r} must be a 64-character hex "
                    f"string, got {digest!r}"
                )
            try:
                int(digest, 16)
            except ValueError as exc:
                raise ValueError(
                    f"expected SHA256 for {key!r} is not hexadecimal: {digest!r}"
                ) from exc
            expected_norm[key] = digest.lower()

    # -- local target mapping and boundary checks ---------------------------
    local_targets = {}
    seen_local = set()
    for key in keys:
        target_path = _safe_local_path(destination_abs, key)
        if target_path in seen_local:
            raise ValueError(
                f"logical keys {key!r} and another selected key map to the "
                f"same local file {target_path!r}"
            )
        seen_local.add(target_path)
        local_targets[key] = target_path
        if target_path == checkpoint_abs:
            raise ValueError(
                "checkpoint path must not be the same file as a selected "
                f"target file: {target_path!r}"
            )

    for key, target_path in local_targets.items():
        rel = os.path.relpath(target_path, destination_abs).split(os.sep)
        _check_no_symlink_escape(destination_abs, rel[:-1], "destination")
        if os.path.islink(target_path):
            raise ReferenceMaterializationError(
                key,
                f"target file {target_path!r} is an existing symlink; refusing "
                "to write through a symlink",
            )

    # -- resolve references for selected keys only --------------------------
    plan = {}
    for key in keys:
        plan[key] = await _resolve_entry(fs, key)

    fingerprint = _compute_fingerprint(
        destination_abs, fs.target, expected_norm, plan
    )

    # -- checkpoint handling -------------------------------------------------
    if resume:
        state = _load_checkpoint(checkpoint_abs)
        _validate_checkpoint_structure(state)
        if state["destination"] != destination_abs:
            raise ReferenceMaterializationError(
                None,
                "checkpoint destination "
                f"{state['destination']!r} does not match current destination "
                f"{destination_abs!r}; refusing to reuse results",
            )
        if state["target"] != fs.target:
            raise ReferenceMaterializationError(
                None,
                "the default target recorded in the checkpoint "
                f"({state['target']!r}) differs from the current default "
                f"target ({fs.target!r})"
            )
        if state["expected_sha256"] != expected_norm:
            raise ReferenceMaterializationError(
                None,
                "expected_sha256 differs from the value frozen in the "
                "checkpoint; refusing to resume against a different plan"
            )
        if set(state["plan"]) != set(keys):
            raise ReferenceMaterializationError(
                None,
                "the selected key set differs from the key set frozen in the "
                "checkpoint; refusing to resume against a different plan"
            )
        for key in keys:
            if _entry_identity(state["plan"][key]) != _entry_identity(plan[key]):
                raise ReferenceMaterializationError(
                    None,
                    f"resolved reference for {key!r} differs from the reference "
                    "frozen in the checkpoint; refusing to resume against a "
                    "different plan"
                )
        current_fingerprint = _compute_fingerprint(
            destination_abs, fs.target, expected_norm, plan
        )
        if state["fingerprint"] != current_fingerprint:
            raise ReferenceMaterializationError(
                None,
                "plan fingerprint mismatch: the current source references, "
                "destination root or expected digests do not match the frozen "
                "checkpoint"
            )
        # Share the mapping itself so every completion update is serialized
        # by subsequent checkpoint writes.
        completed = state["completed"]
    else:
        if os.path.lexists(checkpoint_abs):
            raise ValueError(
                f"checkpoint already exists: {checkpoint_abs!r}; pass resume=True "
                "to continue an existing materialization"
            )
        for key, target_path in local_targets.items():
            if os.path.lexists(target_path):
                raise ValueError(
                    f"target file already exists for logical key {key!r}: "
                    f"{target_path!r}; refusing to overwrite an unverified file"
                )
        completed = {}
        state = {
            "format": _CHECKPOINT_FORMAT,
            "version": _CHECKPOINT_VERSION,
            "fingerprint": fingerprint,
            "destination": destination_abs,
            "target": fs.target,
            "expected_sha256": expected_norm,
            "plan": plan,
            "completed": completed,
            "order": list(keys),
        }
        # All validation is done: only now may we create directories / files.
        try:
            await asyncio.to_thread(os.makedirs, destination_abs, exist_ok=True)
            ckpt_dir = os.path.dirname(checkpoint_abs) or "."
            await asyncio.to_thread(os.makedirs, ckpt_dir, exist_ok=True)
            await _write_checkpoint(checkpoint_abs, state)
        except ReferenceMaterializationError:
            raise
        except Exception as exc:
            raise ReferenceMaterializationError(
                None, "failed to persist the initial plan checkpoint"
            ) from exc

    # -- worker pool ---------------------------------------------------------
    semaphore = asyncio.Semaphore(max_concurrency)
    checkpoint_lock = asyncio.Lock()
    blocking = _BlockingWork()
    # Redirect asyncio.to_thread / run_in_executor(None) (including the ones
    # inside AsyncFileSystemWrapper around synchronous backends) at our
    # tracked executor for the lifetime of this call.
    blocking.install()
    # Track temp files (and their open descriptors) known to this run so
    # cancellation/error cleanup never touches unrelated files and never
    # leaks descriptors from a cancelled mkstemp await.
    temp_files: set[str] = set()
    open_fds: set[int] = set()
    queue: asyncio.Queue = asyncio.Queue()
    for key in keys:
        queue.put_nowait(key)

    async def worker() -> None:
        while True:
            try:
                key = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await _process_key(key)
            except (ReferenceMaterializationError, asyncio.CancelledError):
                # Already a per-key materialization error, or a cancellation
                # that must propagate with its type intact.
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                raise
            except BaseException as exc:
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                raise ReferenceMaterializationError(
                    key, "local write, rename or checkpoint update failed"
                ) from exc

    async def _process_key(key: str) -> None:
        target_path = local_targets[key]
        entry = plan[key]
        rel = os.path.relpath(target_path, destination_abs).split(os.sep)

        # Resume path: trust nothing but the actual bytes on disk.
        if key in completed:
            if os.path.islink(target_path) or not os.path.isfile(target_path):
                # Missing or replaced-by-symlink target: redo it. The plan
                # established ownership of this path.
                completed.pop(key, None)
            else:
                size, digest = await _stream_file_sha256(target_path, blocking)
                rec = state["completed"][key]
                ok = size == rec["size"] and digest == rec["sha256"]
                if ok and expected_norm is not None:
                    ok = digest == expected_norm[key]
                if ok:
                    return
                completed.pop(key, None)

        # Re-verify containment right before writing (defense in depth; the
        # caller is trusted to own the directory, so no full TOCTOU model).
        _check_no_symlink_escape(destination_abs, rel[:-1], "destination")
        parent = os.path.dirname(target_path)
        await _to_thread(blocking, os.makedirs, parent, exist_ok=True)

        fd, tmp_path = await _to_thread(
            blocking, _mkstemp_tracked, temp_files, open_fds, parent
        )
        hasher = hashlib.sha256()
        written = 0
        f = None
        try:
            f = os.fdopen(fd, "wb")
            # The descriptor is now owned by the file object.
            open_fds.discard(fd)

            def _write_chunk(chunk: bytes) -> None:
                if _WRITE_FAILURE_HOOK is not None:
                    raise _WRITE_FAILURE_HOOK
                f.write(chunk)
                hasher.update(chunk)

            if entry["kind"] == "inline":
                payload = base64.b64decode(entry["inline_b64"])
                # Inline payload already resides in memory and is explicitly
                # excluded from the remote-payload memory budget.
                await _to_thread(blocking, _write_chunk, payload)
                written += len(payload)
            else:
                backend = fs.fss[entry["protocol"]]
                url = entry["url"]
                offset = entry["offset"]
                size = entry["size"]
                logical_pos = 0
                while logical_pos < size:
                    request_size = min(block_size, size - logical_pos)
                    if offset is None:
                        start = logical_pos
                        end = logical_pos + request_size
                    else:
                        start = offset + logical_pos
                        end = offset + logical_pos + request_size
                    async with semaphore:
                        try:
                            chunk = await backend._cat_file(
                                url, start=start, end=end
                            )
                        except Exception as exc:
                            raise ReferenceMaterializationError(
                                key,
                                f"remote read failed for {url!r} range "
                                f"[{start}, {end})"
                            ) from exc
                    if not isinstance(chunk, (bytes, bytearray)):
                        raise ReferenceMaterializationError(
                            key,
                            f"backend returned non-bytes payload "
                            f"({type(chunk).__name__}) for {url!r}",
                        )
                    if len(chunk) != request_size:
                        raise ReferenceMaterializationError(
                            key,
                            f"short read for {url!r} range [{start}, {end}): "
                            f"expected {request_size} bytes, got {len(chunk)}",
                        )
                    # Never request or write beyond the referenced range.
                    await _to_thread(blocking, _write_chunk, bytes(chunk))
                    written += len(chunk)
                    logical_pos += len(chunk)

            await _to_thread(blocking, f.flush)
            await _to_thread(blocking, os.fsync, f.fileno())
            await _to_thread(blocking, f.close)
            f = None

            if written != entry["size"]:
                raise ReferenceMaterializationError(
                    key,
                    f"length mismatch: wrote {written} bytes but the reference "
                    f"plan requires {entry['size']} bytes",
                )
            digest = hasher.hexdigest()
            if expected_norm is not None and digest != expected_norm[key]:
                raise ReferenceMaterializationError(
                    key,
                    f"sha256 mismatch: expected {expected_norm[key]}, got {digest}",
                )

            # Rename can succeed while the process dies before the checkpoint
            # update; on resume the file is not marked completed so it is
            # re-materialized and the record rebuilt.
            def _replace() -> None:
                if _RENAME_FAILURE_HOOK is not None:
                    raise _RENAME_FAILURE_HOOK
                os.replace(tmp_path, target_path)
                real = os.path.realpath(target_path)
                if os.path.commonpath([real, destination_abs]) != destination_abs:
                    raise ReferenceMaterializationError(
                        key,
                        f"final target {target_path!r} resolves outside the "
                        "destination root; refusing to record it",
                    )

            await _to_thread(blocking, _replace)
            temp_files.discard(tmp_path)

            async with checkpoint_lock:
                completed[key] = {"size": written, "sha256": digest}
                try:
                    await _write_checkpoint(checkpoint_abs, state)
                except ReferenceMaterializationError:
                    raise
                except Exception as exc:
                    raise ReferenceMaterializationError(
                        key, "failed to persist the progress checkpoint"
                    ) from exc
            payload_bytes_delta[key] = written
        except BaseException:
            if f is not None:
                try:
                    await _to_thread(blocking, f.close)
                except Exception as exc:  # cleanup must not mask the real error
                    logger.warning("error closing temp file %s: %s", tmp_path, exc)
            elif fd in open_fds:
                # Cancelled before the descriptor was wrapped in a file object.
                try:
                    await _to_thread(blocking, os.close, fd)
                except OSError as exc:
                    logger.warning("error closing fd %s: %s", fd, exc)
                open_fds.discard(fd)
            try:
                await _to_thread(blocking, os.unlink, tmp_path)
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.warning(
                    "failed to remove temp file %s: %s", tmp_path, exc
                )
            temp_files.discard(tmp_path)
            raise

    payload_bytes_delta: dict[str, int] = {}
    worker_count = min(max_concurrency, len(keys))
    worker_tasks = [
        asyncio.ensure_future(worker()) for _ in range(worker_count)
    ]

    async def _shutdown() -> None:
        # On first failure or external cancellation, no new key is claimed
        # (the failing worker drained the queue); cancel any still-running
        # workers and wait for *all* of them to finish before propagating.
        for task in worker_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*worker_tasks, return_exceptions=True)
        # Wrapped synchronous backends cannot be interrupted mid-call; wait
        # for all blocking work issued through the run-scoped executor.
        await blocking.wait_all()
        # Close descriptors from mkstemp calls whose awaiting worker was
        # cancelled before it could wrap or clean up.
        for fd in list(open_fds):
            try:
                await _to_thread(blocking, os.close, fd)
            except OSError as exc:
                logger.warning("failed to close temp fd %s: %s", fd, exc)
            open_fds.discard(fd)
        # Best-effort removal of temp files still known to this run.
        for tmp_path in list(temp_files):
            try:
                await _to_thread(blocking, os.unlink, tmp_path)
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.warning(
                    "failed to remove temp file %s: %s", tmp_path, exc
                )
            temp_files.discard(tmp_path)

    try:
        try:
            await asyncio.gather(*worker_tasks)
        except BaseException:
            # The shield keeps the shutdown sequence alive even when an
            # external cancellation is already pending on this coroutine.
            await asyncio.shield(_shutdown())
            raise
    finally:
        # Restore the loop's previous default executor in every outcome.
        blocking.restore()

    # -- build result in this call's input order ----------------------------
    written = []
    reused = []
    for key in keys:
        if key in payload_bytes_delta:
            written.append(key)
        else:
            reused.append(key)
    return {
        "written": written,
        "reused": reused,
        "bytes_written": sum(payload_bytes_delta.values()),
        "checkpoint": checkpoint_abs,
    }


async def _write_checkpoint(checkpoint_path: str, state: dict) -> None:
    """Atomically persist the progress JSON via a same-directory temp file."""
    data = (json.dumps(state, sort_keys=True, indent=2) + "\n").encode("utf-8")
    await asyncio.to_thread(_atomic_write, checkpoint_path, data)


def materialize_sync(fs, *args, **kwargs):
    """Synchronous entry point used by ReferenceFileSystem.materialize."""
    return sync(fs.loop, materialize, fs, *args, **kwargs)
