"""Synchronous, fully local example of ReferenceFileSystem.materialize.

Run it directly:

    python examples/materialize_sync_example.py

Everything runs against local temporary directories and the local file
backend (which fsspec wraps in an async adapter internally): no network,
credentials or external services are used. The script builds a small
reference set containing inline data, a base64 inline value, a whole-file
reference and a byte-range reference, materializes it, simulates a crash,
and then resumes from the checkpoint.
"""

import hashlib
import os
import tempfile

import fsspec


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    workdir = tempfile.mkdtemp(prefix="materialize-sync-")
    remote_dir = os.path.join(workdir, "remote")
    dest = os.path.join(workdir, "out")
    checkpoint = os.path.join(workdir, "progress", "checkpoint.json")
    os.makedirs(remote_dir)

    # A "remote" payload served by the ordinary local filesystem.
    big = bytes((i * 7) % 256 for i in range(2500))
    with open(os.path.join(remote_dir, "big.bin"), "wb") as f:
        f.write(big)
    big_url = "file://" + os.path.join(remote_dir, "big.bin")

    refs = {
        # plain inline value
        "meta/inline.txt": b"hello from an inline reference",
        # base64-prefixed inline value
        "meta/b64.txt": b"base64:" + __import__("base64").b64encode(b"base64!"),
        # whole-file reference
        "data/whole.bin": [big_url],
        # byte-range reference: logical bytes map to big.bin[100:600]
        "data/range.bin": [big_url, 100, 500],
        # zero-length reference
        "data/empty.bin": [big_url, 0, 0],
    }
    fs = fsspec.filesystem(
        "reference",
        fo=refs,
        remote_protocol="file",
        skip_instance_cache=True,
    )

    expected = {
        "meta/inline.txt": sha256_hex(b"hello from an inline reference"),
        "meta/b64.txt": sha256_hex(b"base64!"),
        "data/whole.bin": sha256_hex(big),
        "data/range.bin": sha256_hex(big[100:600]),
        "data/empty.bin": sha256_hex(b""),
    }

    print(f"workdir: {workdir}")
    result = fs.materialize(
        ["meta/inline.txt", "meta/b64.txt", "data/whole.bin",
         "data/range.bin", "data/empty.bin"],
        dest,
        checkpoint=checkpoint,
        block_size=512,       # at most 512 bytes per remote request
        max_concurrency=2,    # at most 2 remote reads in flight
        expected_sha256=expected,
    )
    print("first run:", result)
    for root, _dirs, files in os.walk(dest):
        for name in sorted(files):
            path = os.path.join(root, name)
            print("  published:", os.path.relpath(path, dest),
                  os.stat(path).st_size, "bytes")

    # Simulate the rename/checkpoint-update window: leave a valid target file
    # on disk while removing its "completed" record, then resume. The resume
    # re-verifies every completed file by streaming its bytes, redoes the
    # unmarked file, and never re-downloads files that still verify.
    import json
    with open(checkpoint, encoding="utf-8") as f:
        state = json.load(f)
    state["completed"].pop("data/range.bin")
    tmp_checkpoint = checkpoint + ".tmp"
    with open(tmp_checkpoint, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp_checkpoint, checkpoint)

    resumed = fs.materialize(
        ["data/range.bin", "data/whole.bin", "meta/inline.txt",
         "meta/b64.txt", "data/empty.bin"],  # reordered on purpose
        dest,
        checkpoint=checkpoint,
        block_size=1024,      # block_size may change between runs
        max_concurrency=3,    # concurrency may change between runs
        expected_sha256=expected,  # frozen as part of the plan fingerprint
        resume=True,
    )
    print("resume:  ", resumed)
    assert resumed["reused"] == [
        "data/whole.bin", "meta/inline.txt", "meta/b64.txt", "data/empty.bin"
    ]
    assert resumed["written"] == ["data/range.bin"]
    assert resumed["bytes_written"] == 500
    with open(os.path.join(dest, "data/range.bin"), "rb") as f:
        assert f.read() == big[100:600]
    print("OK: resume verified bytes on disk and only redid the unmarked file")


if __name__ == "__main__":
    main()
