# Local reproducible evaluation environment

Source: https://github.com/fsspec/filesystem_spec, tag `2025.10.0`.
The annotated tag resolves to commit
`dcb167e8f50e6273d4cfdfc4cab8fc5aa4c958bf` (BSD-3-Clause).
This evaluation uses synthetic local data, in-memory filesystem backends and
local loopback HTTP fixtures; it does not represent private production data.

The fixed Linux ARM64 evaluation image supplies Python 3.11.2, but neither pip
nor ensurepip. Install into the container user home, not the host Python:

```sh
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/fsspec-get-pip.py
python3 /tmp/fsspec-get-pip.py --user --break-system-packages pip==24.3.1
python3 -m pip install --user --break-system-packages -r requirements-cc-solo.txt
SETUPTOOLS_SCM_PRETEND_VERSION=2025.10.0 python3 -m pip install --user --break-system-packages --no-build-isolation --no-deps -e .
python3 -m pytest -q -rs -o asyncio_default_fixture_loop_scope=function fsspec/implementations/tests/test_reference.py fsspec/tests/test_async.py
```

`requirements-cc-solo.in` records the direct dependency selection;
`requirements-cc-solo.txt` records the resolved pinned installation. Editable
installation generates the ignored `fsspec/_version.py` build artifact.
Optional cloud, FTP/SSH, parquet/pandas, and downstream integrations have
separate dependencies and are not part of this minimal setup. Tests must state
which local behaviors were actually checked and which integrations were skipped.
No external cloud account, credentials, real datasets or external service is
required for the selected reference filesystem and asyncio tests.

The build-only version override makes the package version reproducible even in
the runner's shallow initial-commit checkout, which may not include the source
tag. It does not change the evaluated source, model settings or tool entrypoint.
The original async scheduler tests can finish with passing assertions while
also emitting never-awaited/unretrieved-task warnings on their intentional
error paths. Preserve and inspect those warnings separately from assertion
results. Optional parquet/zarr/kerchunk integration skips are not verification
of those integrations.
