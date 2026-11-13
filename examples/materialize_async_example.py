"""Asynchronous, fully local example of ReferenceFileSystem._materialize.

Run it directly:

    python examples/materialize_async_example.py

This uses an in-process async file-system double backed by a real temporary
directory. The double records every byte-range request and the observed
concurrency peak, demonstrating that:

* no remote request exceeds ``block_size``;
* no more than ``max_concurrency`` payload reads run at once;
* the coroutine can be awaited directly on an asynchronous
  ReferenceFileSystem without nesting ``asyncio.run``;
* cancelling a slow run waits for in-flight work to settle and the run can
  then be resumed from the checkpoint.
"""

import asyncio
import hashlib
import os
import pathlib
import tempfile

import fsspec
from fsspec.asyn import AsyncFileSystem


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class LocalRecordingAsyncFS(AsyncFileSystem):
    """Minimal local async backend: records ranges and concurrency."""

    protocol = "local-recording"
    cachable = False

    def __init__(self, root, delay=0.0, **kwargs):
        super().__init__(**kwargs)
        self.root = pathlib.Path(root)
        self.delay = delay
        self.requests = []
        self.active = 0
        self.peak = 0
        self._lock = asyncio.Lock()

    def _disk(self, path):
        name = path.split("://", 1)[1] if "://" in path else path
        return self.root / name.lstrip("/")

    async def _info(self, path, **kwargs):
        disk = self._disk(path)
        return {"name": path, "size": disk.stat().st_size, "type": "file"}

    async def _cat_file(self, path, start=None, end=None, **kwargs):
        disk = self._disk(path)
        async with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)

            def read():
                with open(disk, "rb") as f:
                    if start is not None:
                        f.seek(start)
                    return f.read() if end is None else f.read(end - start)

            data = await asyncio.get_running_loop().run_in_executor(None, read)
            self.requests.append((path, start, end))
            return data
        finally:
            async with self._lock:
                self.active -= 1


async def main() -> None:
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="materialize-async-"))
    remote = workdir / "remote"
    remote.mkdir()
    payload = bytes((i * 3 + 1) % 256 for i in range(4000))
    (remote / "data.bin").write_bytes(payload)

    backend = LocalRecordingAsyncFS(remote, asynchronous=True)
    refs = {f"chunk-{i:02d}": ["local-recording://data.bin", i * 100, 100]
            for i in range(10)}
    fs = fsspec.filesystem(
        "reference", fo=refs, fs=backend, asynchronous=True,
        skip_instance_cache=True,
    )

    dest = workdir / "out"
    checkpoint = workdir / "progress.json"
    keys = list(refs)
    result = await fs._materialize(
        keys,
        str(dest),
        checkpoint=str(checkpoint),
        block_size=64,
        max_concurrency=3,
        expected_sha256={
            key: sha256_hex(payload[i * 100:i * 100 + 100])
            for i, key in enumerate(keys)
        },
    )
    print("written:", result["written"][:3], "...")
    print("bytes_written:", result["bytes_written"])
    print("backend requests:", len(backend.requests),
          "max request size:", max(e - s for _, s, e in backend.requests),
          "concurrency peak:", backend.peak)
    assert max(e - s for _, s, e in backend.requests) <= 64
    assert backend.peak <= 3

    # A second, delayed backend demonstrates cancellation + resume.
    slow = LocalRecordingAsyncFS(remote, delay=0.05, asynchronous=True)
    slow_fs = fsspec.filesystem(
        "reference", fo=refs, fs=slow, asynchronous=True,
        skip_instance_cache=True,
    )
    task = asyncio.ensure_future(
        slow_fs._materialize(
            keys,
            str(workdir / "out2"),
            checkpoint=str(workdir / "progress2.json"),
            block_size=64,
            max_concurrency=3,
        )
    )
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("cancelled slow run; active backend reads now:", slow.active)
    assert slow.active == 0

    resumed = await slow_fs._materialize(
        keys,
        str(workdir / "out2"),
        checkpoint=str(workdir / "progress2.json"),
        block_size=128,
        max_concurrency=2,
        resume=True,
    )
    print("resume written:", len(resumed["written"]),
          "reused:", len(resumed["reused"]))
    assert sorted(resumed["written"] + resumed["reused"]) == sorted(keys)
    for i, key in enumerate(keys):
        assert (workdir / "out2" / key).read_bytes() == payload[
            i * 100:i * 100 + 100
        ]
    print("OK: all materialized files match their referenced ranges")
    print("workdir:", workdir)


if __name__ == "__main__":
    asyncio.run(main())
