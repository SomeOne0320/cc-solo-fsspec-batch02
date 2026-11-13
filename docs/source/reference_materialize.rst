.. _materialize:

Resumable reference materialization (``materialize``)
======================================================

.. currentmodule:: fsspec.implementations.reference

:meth:`ReferenceFileSystem.materialize` materializes selected *logical*
reference keys into ordinary local files. It is an independent interface
alongside ``get``/``cat``/``open`` — none of those existing methods or their
default behavior change when this feature is present.

Why a separate interface?
-------------------------

The existing bulk entry points are not suitable for an offline, resumable ETL
step:

* :meth:`ReferenceFileSystem.get_file` first reads an entire logical file into
  memory before writing it locally;
* :meth:`ReferenceFileSystem.get` collects every selected payload into memory
  via ``cat`` before writing anything;
* neither one records verifiable progress, so an interrupted run cannot tell
  which files are complete and trustworthy.

``materialize`` instead:

* streams every logical file in fixed-size chunks (never more than
  ``block_size`` bytes per remote request, never more than
  ``max_concurrency`` payload reads in flight);
* writes each file through a same-directory temporary file and only publishes
  it with an atomic ``os.replace`` after length and SHA256 verification;
* persists a versioned JSON checkpoint **before** any payload work starts and
  updates it atomically after every completed file;
* supports resuming from the last successfully replaced checkpoint, verifying
  already completed files by re-reading their actual bytes rather than
  trusting their existence.

Usage
-----

Synchronous (blocking) call, from ordinary code without a running event loop::

    result = fs.materialize(
        ["key/a", "key/b", "key/c"],
        "/data/out",
        checkpoint="/data/progress/materialize.json",
        block_size=1 << 20,
        max_concurrency=4,
    )
    # result == {
    #     "written": ["key/a", "key/b", "key/c"],
    #     "reused": [],
    #     "bytes_written": 12345,
    #     "checkpoint": "/data/progress/materialize.json",
    # }

Asynchronous call — on an async instance you can ``await`` the private
async entry point directly; no ``asyncio.run`` is nested inside a running
loop::

    result = await fs._materialize(
        ["key/a", "key/b"],
        "/data/out",
        checkpoint="/data/progress/materialize.json",
        resume=True,
    )

The synchronous method uses fsspec's dedicated IO-loop thread
(:func:`fsspec.asyn.sync`); it never calls ``asyncio.run`` itself, so it does
not nest event loops. A normal (``asynchronous=False``) instance owns an IO
loop on a separate thread, so its synchronous method can even be called from
code that has its own running loop. Calling a synchronous method on an
``asynchronous=True`` instance from that instance's own running loop raises
the standard ``NotImplementedError("Calling sync() from within a running
loop")`` — use ``await fs._materialize(...)`` there instead.

Inputs and validation
---------------------

``paths`` is one logical key string or a sequence of them. The given order is
preserved in the returned ``written``/``reused`` lists. The interface performs
**no** glob expansion and **no** recursive expansion. The following are all
rejected before any target data or temporary file is created:

* duplicate logical keys (they are never silently de-duplicated);
* absolute keys, empty segments, ``.``/``..`` segments and backslashes;
* file/directory prefix conflicts, e.g. selecting both ``a`` and ``a/b``;
* two keys mapping to the same local file;
* any existing symlink on the output path that would let writes escape the
  destination root (the final real path is also re-checked after rename);
* a checkpoint path identical to one of the selected target files;
* ``block_size`` / ``max_concurrency`` that are not positive integers —
  booleans are rejected even though ``bool`` is an ``int`` subclass.

``expected_sha256`` must be either ``None`` or a mapping covering **exactly**
the selected keys, each value a 64-character hexadecimal digest. Upper- and
lower-case hex are both accepted; digests are normalized to lower case for
both comparison and the plan fingerprint.

Reference semantics
-------------------

Reference resolution reuses :meth:`ReferenceFileSystem._cat_common`, so every
shape already accepted by ``cat``/``open`` is supported identically:

* spec V0 dictionaries and spec V1 references with rendered templates
  (simple and Jinja);
* the instance default ``target`` for references whose URL is ``None``;
* inline values given as raw bytes/strings and ``base64:``-prefixed values;
* whole-file references ``[url]`` and byte-range references
  ``[url, offset, size]``, including zero-length ranges and zero-length inline
  values.

For a whole-file reference the backend's async ``_info`` must report a
non-negative integer ``size``. An unknown or non-integer size is rejected with
a :class:`ReferenceMaterializationError`; the implementation never falls back
to reading an unbounded whole file with ``cat``.

Range requests are issued relative to the logical file offsets inside the
reference. Every request stays within the referenced byte range; nothing
outside it is ever read or written. If a backend returns fewer bytes than
were requested, that is a *short read* error: the truncated content is never
treated as success, the temporary file is removed and the logical key fails
(see below). Inline payloads already resident in the reference set are not
part of the remote-payload memory budget and do not acquire a concurrency
slot.

Memory and concurrency bounds
-----------------------------

* each remote request asks for at most ``block_size`` bytes;
* at most ``max_concurrency`` remote payload reads are active at any moment
  (an :class:`asyncio.Semaphore` guards the backend ``_cat_file`` calls);
* additional remote payload resident in memory is therefore
  ``O(max_concurrency * block_size)``;
* plan and progress metadata is ``O(number of selected keys)``;
* the implementation never calls ``cat``/``get`` to pre-collect a whole file
  or the whole selection, and only references for the selected keys are
  resolved.

Local file publication
----------------------

For every key the implementation:

1. creates parent directories as needed (after all upfront validation);
2. opens a unique temporary file in the **same directory** as the final
   target (prefix ``.fsspec-materialize-``);
3. streams chunks into it while incrementally computing SHA256;
4. flushes, ``fsync``s and closes it;
5. verifies the written length against the plan and the digest against
   ``expected_sha256`` when provided;
6. atomically replaces the final path with ``os.replace`` and re-checks that
   the resolved real path stays inside the destination root;
7. records ``{"size", "sha256"}`` for the key in the checkpoint, itself
   written via a same-directory temporary file and atomic replace.

Only complete, verified files ever appear at a final path. A failure while
writing or verifying a file removes its temporary file and never publishes
it. Files of other keys that already completed and were recorded are kept:
materialization is deliberately **not** an all-or-nothing transaction across
files. Unrelated files already present in ``destination`` are never touched.

Checkpoint format and the progress contract
-------------------------------------------

The checkpoint is pretty-printed JSON with this top-level shape (version 1)::

    {
      "format": "fsspec-reference-materialization",
      "version": 1,
      "fingerprint": "<sha256 hex>",
      "destination": "<absolute, normalized destination root>",
      "target": "<ReferenceFileSystem.target used for None URLs>",
      "expected_sha256": {"<key>": "<64 hex>"},   // or null
      "order": ["<key>", ...],                    // first-run input order
      "plan": {
        "<key>": {
          "kind": "inline",
          "inline_b64": "<base64 payload>",
          "size": 123
        }
      },
      "completed": {
        "<key>": {"size": 123, "sha256": "<64 hex>"}
      }
    }

URL plan entries additionally bind ``url``, ``offset`` (``null`` for whole
files), ``size`` and the ``default_target`` flag, so the resolved identity of
every reference — inline content or URL/range/default-target choice — is
frozen. The ``fingerprint`` is a SHA256 over canonical (sorted,
separator-minimal) JSON of the normalized destination, the default target,
the normalized expected digests and every plan entry's identity.

First run (``resume=False``, the default)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* an existing checkpoint file is rejected;
* any already existing selected target file is rejected — an unverified
  pre-existing file is never overwritten;
* all reference resolution, parameter validation, expected-digest and
  target-conflict checks finish before any directory, target file or payload
  temporary file is created;
* a checkpoint containing the complete plan is atomically saved **before**
  payload work starts.

Resume (``resume=True``)
^^^^^^^^^^^^^^^^^^^^^^^^

* the checkpoint must exist; missing file is a ``ValueError``;
* bad JSON, an unknown format/version, missing fields, inconsistent
  order/plan/completed structures or malformed digests raise a
  :class:`ReferenceMaterializationError` with ``path=None`` — a corrupt
  checkpoint is never treated as an empty progress file;
* the frozen destination root, default target, expected digests, selected key
  set and every resolved reference identity must match the current call, and
  the stored fingerprint must match the freshly computed one. Otherwise the
  run is refused before any result is reused or overwritten;
* ``block_size``, ``max_concurrency`` and the **order** of ``paths`` may
  change without invalidating the same plan — output lists still follow the
  current call's order;
* every key recorded as ``completed`` is re-checked by streaming the actual
  local file through SHA256: the real length and digest must match the
  record, and the digest must match ``expected_sha256`` when given. Only then
  is the key counted in ``reused`` and its network read skipped;
* a missing, symlinked or corrupt (wrong length or digest, including a
  same-size corrupt file) target is re-materialized. Existence or size alone
  is never sufficient.

The rename/checkpoint-update window
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If ``os.replace`` succeeds but the process dies before the checkpoint update,
the file exists on disk without a ``completed`` record. On resume it is not
trusted: the key is re-materialized (the plan already established ownership of
that output path) and the record is rebuilt.

What the fingerprint does *not* freeze
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The fingerprint binds the *reference plan* — the inline contents and the
URL/offset/size/default-target tuples — not the bytes currently served by a
remote URL. Without ``expected_sha256`` there is no remote snapshot
guarantee: if the contents behind an unchanged URL change between runs, a
resume cannot detect that. Provide a complete ``expected_sha256`` map when
snapshot consistency is required.

Errors and cancellation
-----------------------

Per-key failures (reference resolution, remote reads, short reads, local
writes, digest mismatches, rename or checkpoint updates) raise
:class:`ReferenceMaterializationError` whose ``path`` attribute is the
offending logical key (``None`` for global plan/checkpoint validation
failures). The original reason is retained via exception chaining. Global
parameter mistakes may also be raised as plain ``ValueError``/``TypeError``.

After the first failure no worker claims a new key; all workers created by
the call are cancelled if still running and **awaited** before the error
propagates. No background task keeps writing after the call returns. On
external cancellation the same shutdown runs and the original
:class:`asyncio.CancelledError` propagates unchanged — it is never wrapped as
a materialization failure.

Backends that are natively async have their in-flight ``_cat_file`` calls
cancelled normally. Wrapped **synchronous** backends execute blocking calls in
worker threads; a running blocking call cannot be interrupted. The
materialization call nevertheless waits for every blocking call it already
issued to finish before returning (and before propagating cancellation), then
cleans up. Temporary files and file handles owned by the run are closed and
removed on every error/cancellation path; unrelated files are never deleted
and source data is never modified. Cleanup errors are logged as warnings and
do not mask the original business exception.

Known limitations
-----------------

* Recovery relies only on the most recently atomically replaced checkpoint.
  Power-loss atomicity of the underlying filesystem, cross-process locking and
  automatic reaping of unknown orphan temporary files left by ``SIGKILL`` are
  out of scope; the caller is expected to own the output directory and
  checkpoint file exclusively for the duration of the run.
* Defense against another concurrently running process mutating the same
  local paths (TOCTOU races) is not attempted.
* Remote content drift behind an unchanged URL is only detectable when
  ``expected_sha256`` is supplied.
