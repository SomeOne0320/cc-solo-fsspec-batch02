import json
import os

import pytest

import fsspec
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.reference import (
    LazyReferenceMapper,
    ReferenceFileSystem,
    ReferenceNotReachable,
)
from fsspec.tests.conftest import data, reset_files, server, win  # noqa: F401


def test_simple(server):
    # The dictionary in refs may be dumped with a different separator
    # depending on whether json or ujson is imported
    from fsspec.implementations.reference import json as json_impl

    refs = {
        "a": b"data",
        "b": (server.realfile, 0, 5),
        "c": (server.realfile, 1, 5),
        "d": b"base64:aGVsbG8=",
        "e": {"key": "value"},
    }
    h = fsspec.filesystem("http")
    fs = fsspec.filesystem("reference", fo=refs, fs=h)

    assert fs.cat("a") == b"data"
    assert fs.cat("b") == data[:5]
    assert fs.cat("c") == data[1 : 1 + 5]
    assert fs.cat("d") == b"hello"
    assert fs.cat("e") == json_impl.dumps(refs["e"]).encode("utf-8")
    with fs.open("d", "rt") as f:
        assert f.read(2) == "he"


def test_open(m):
    from fsspec.implementations.reference import json as json_impl

    m.pipe("/data/0", data)
    refs = {
        "a": b"data",
        "b": ["memory://data/0"],
        "c": ("memory://data/0", 0, 5),
        "d": ("memory://data/0", 1, 5),
        "e": b"base64:aGVsbG8=",
        "f": {"key": "value"},
    }
    fs = fsspec.filesystem("reference", fo=refs, fs=m)

    with fs.open("a", "rb") as f:
        assert f.read() == b"data"

    with fs.open("b", "rb") as f:
        assert f.read() == data

    with fs.open("c", "rb") as f:
        assert f.read() == data[:5]
        assert not f.read()

    with fs.open("d", "rb") as f:
        assert f.read() == data[1:6]
        assert not f.read()

    with fs.open("e", "rb") as f:
        assert f.read() == b"hello"

    with fs.open("f", "rb") as f:
        assert f.read() == json_impl.dumps(refs["f"]).encode("utf-8")

    # check partial reads
    with fs.open("c", "rb") as f:
        assert f.read(2) == data[:2]
        f.seek(2, os.SEEK_CUR)
        assert f.read() == data[4:5]

    with fs.open("d", "rb") as f:
        assert f.read(2) == data[1:3]
        f.seek(1, os.SEEK_CUR)
        assert f.read() == data[4:6]


def test_simple_ver1(server):
    # The dictionary in refs may be dumped with a different separator
    # depending on whether json or ujson is imported
    from fsspec.implementations.reference import json as json_impl

    in_data = {
        "version": 1,
        "refs": {
            "a": b"data",
            "b": (server.realfile, 0, 5),
            "c": (server.realfile, 1, 5),
            "d": b"base64:aGVsbG8=",
            "e": {"key": "value"},
        },
    }
    h = fsspec.filesystem("http")
    fs = fsspec.filesystem("reference", fo=in_data, fs=h)

    assert fs.cat("a") == b"data"
    assert fs.cat("b") == data[:5]
    assert fs.cat("c") == data[1 : 1 + 5]
    assert fs.cat("d") == b"hello"
    assert fs.cat("e") == json_impl.dumps(in_data["refs"]["e"]).encode("utf-8")
    with fs.open("d", "rt") as f:
        assert f.read(2) == "he"


def test_target_options(m):
    m.pipe("data/0", b"hello")
    refs = {"a": ["memory://data/0"]}
    fn = "memory://refs.json.gz"
    with fsspec.open(fn, "wt", compression="gzip") as f:
        json.dump(refs, f)

    fs = fsspec.filesystem("reference", fo=fn, target_options={"compression": "gzip"})
    assert fs.cat("a") == b"hello"


def test_ls(server):
    refs = {"a": b"data", "b": (server.realfile, 0, 5), "c/d": (server.realfile, 1, 6)}
    h = fsspec.filesystem("http")
    fs = fsspec.filesystem("reference", fo=refs, fs=h)

    assert fs.ls("", detail=False) == ["a", "b", "c"]
    assert {"name": "c", "type": "directory", "size": 0} in fs.ls("", detail=True)
    assert fs.find("") == ["a", "b", "c/d"]
    assert fs.find("", withdirs=True) == ["a", "b", "c", "c/d"]
    assert fs.find("c", detail=True) == {
        "c/d": {"name": "c/d", "size": 6, "type": "file"}
    }


def test_nested_dirs_ls():
    # issue #1430
    refs = {"a": "A", "B/C/b": "B", "B/C/d": "d", "B/_": "_"}
    fs = fsspec.filesystem("reference", fo=refs)
    assert len(fs.ls("")) == 2
    assert {e["name"] for e in fs.ls("")} == {"a", "B"}
    assert len(fs.ls("B")) == 2
    assert {e["name"] for e in fs.ls("B")} == {"B/C", "B/_"}


def test_info(server):
    refs = {
        "a": b"data",
        "b": (server.realfile, 0, 5),
        "c/d": (server.realfile, 1, 6),
        "e": (server.realfile,),
    }
    h = fsspec.filesystem("http", headers={"give_length": "true", "head_ok": "true"})
    fs = fsspec.filesystem("reference", fo=refs, fs=h)
    assert fs.size("a") == 4
    assert fs.size("b") == 5
    assert fs.size("c/d") == 6
    assert fs.info("e")["size"] == len(data)


def test_mutable(server, m):
    refs = {
        "a": b"data",
        "b": (server.realfile, 0, 5),
        "c/d": (server.realfile, 1, 6),
        "e": (server.realfile,),
    }
    h = fsspec.filesystem("http", headers={"give_length": "true", "head_ok": "true"})
    fs = fsspec.filesystem("reference", fo=refs, fs=h)
    fs.rm("a")
    assert not fs.exists("a")

    bin_data = b"bin data"
    fs.pipe("aa", bin_data)
    assert fs.cat("aa") == bin_data

    fs.save_json("memory://refs.json")
    assert m.exists("refs.json")

    fs = fsspec.filesystem("reference", fo="memory://refs.json", remote_protocol="http")
    assert not fs.exists("a")
    assert fs.cat("aa") == bin_data


def test_put_get(tmpdir):
    d1 = f"{tmpdir}/d1"
    os.mkdir(d1)
    with open(f"{d1}/a", "wb") as f:
        f.write(b"1")
    with open(f"{d1}/b", "wb") as f:
        f.write(b"2")
    d2 = f"{tmpdir}/d2"

    fs = fsspec.filesystem("reference", fo={}, remote_protocol="file")
    fs.put(d1, "out", recursive=True)

    fs.get("out", d2, recursive=True)
    assert open(f"{d2}/a", "rb").read() == b"1"
    assert open(f"{d2}/b", "rb").read() == b"2"


def test_put_get_single(tmpdir):
    d1 = f"{tmpdir}/f1"
    d2 = f"{tmpdir}/f2"
    with open(d1, "wb") as f:
        f.write(b"1")

    # skip instance cache since this is the same kwargs as previous test
    fs = fsspec.filesystem(
        "reference", fo={}, remote_protocol="file", skip_instance_cache=True
    )
    fs.put_file(d1, "out")

    fs.get_file("out", d2)
    assert open(d2, "rb").read() == b"1"
    fs.pipe({"hi": b"data"})
    assert fs.cat("hi") == b"data"


def test_defaults(server):
    refs = {"a": b"data", "b": (None, 0, 5)}
    fs = fsspec.filesystem(
        "reference",
        fo=refs,
        target_protocol="http",
        target=server.realfile,
        remote_protocol="http",
    )

    assert fs.cat("a") == b"data"
    assert fs.cat("b") == data[:5]


jdata = """{
    "metadata": {
        ".zattrs": {
            "Conventions": "UGRID-0.9.0"
        },
        ".zgroup": {
            "zarr_format": 2
        },
        "adcirc_mesh/.zarray": {
            "chunks": [
                1
            ],
            "dtype": "<i4",
            "shape": [
                1
            ],
            "zarr_format": 2
        },
        "adcirc_mesh/.zattrs": {
            "_ARRAY_DIMENSIONS": [
                "mesh"
            ],
            "cf_role": "mesh_topology"
        },
        "adcirc_mesh/.zchunkstore": {
            "adcirc_mesh/0": {
                "offset": 8928,
                "size": 4
            },
            "source": {
                "array_name": "/adcirc_mesh",
                "uri": "https://url"
            }
        }
    },
    "zarr_consolidated_format": 1
}
"""


def test_spec1_expand():
    pytest.importorskip("jinja2")
    from fsspec.implementations.reference import json as json_impl

    in_data = {
        "version": 1,
        "templates": {"u": "server.domain/path", "f": "{{c}}"},
        "gen": [
            {
                "key": "gen_key{{i}}",
                "url": "http://{{u}}_{{i}}",
                "offset": "{{(i + 1) * 1000}}",
                "length": "1000",
                "dimensions": {"i": {"stop": 5}},
            },
            {
                "key": "gen_key{{i}}",
                "url": "http://{{u}}_{{i}}",
                "dimensions": {"i": {"start": 5, "stop": 7}},
            },
        ],
        "refs": {
            "key0": "data",
            "key1": ["http://target_url", 10000, 100],
            "key2": ["http://{{u}}", 10000, 100],
            "key3": ["http://{{f(c='text')}}", 10000, 100],
            "key4": ["http://target_url"],
            "key5": {"key": "value"},
        },
    }
    fs = fsspec.filesystem(
        "reference", fo=in_data, target_protocol="http", simple_templates=False
    )
    assert fs.references == {
        "key0": "data",
        "key1": ["http://target_url", 10000, 100],
        "key2": ["http://server.domain/path", 10000, 100],
        "key3": ["http://text", 10000, 100],
        "key4": ["http://target_url"],
        "key5": json_impl.dumps(in_data["refs"]["key5"]),
        "gen_key0": ["http://server.domain/path_0", 1000, 1000],
        "gen_key1": ["http://server.domain/path_1", 2000, 1000],
        "gen_key2": ["http://server.domain/path_2", 3000, 1000],
        "gen_key3": ["http://server.domain/path_3", 4000, 1000],
        "gen_key4": ["http://server.domain/path_4", 5000, 1000],
        "gen_key5": ["http://server.domain/path_5"],
        "gen_key6": ["http://server.domain/path_6"],
    }


def test_spec1_expand_simple():
    pytest.importorskip("jinja2")
    from fsspec.implementations.reference import json as json_impl

    in_data = {
        "version": 1,
        "templates": {"u": "server.domain/path"},
        "refs": {
            "key0": "base64:ZGF0YQ==",
            "key2": ["http://{{u}}", 10000, 100],
            "key4": ["http://target_url"],
            "key5": {"key": "value"},
        },
    }
    fs = fsspec.filesystem("reference", fo=in_data, target_protocol="http")
    assert fs.references["key2"] == ["http://server.domain/path", 10000, 100]
    fs = fsspec.filesystem(
        "reference",
        fo=in_data,
        target_protocol="http",
        template_overrides={"u": "not.org/p"},
    )
    assert fs.references["key2"] == ["http://not.org/p", 10000, 100]
    assert fs.cat("key0") == b"data"
    assert fs.cat("key5") == json_impl.dumps(in_data["refs"]["key5"]).encode("utf-8")


def test_spec1_gen_variants():
    pytest.importorskip("jinja2")
    with pytest.raises(ValueError):
        missing_length_spec = {
            "version": 1,
            "templates": {"u": "server.domain/path"},
            "gen": [
                {
                    "key": "gen_key{{i}}",
                    "url": "http://{{u}}_{{i}}",
                    "offset": "{{(i + 1) * 1000}}",
                    "dimensions": {"i": {"stop": 2}},
                },
            ],
        }
        fsspec.filesystem("reference", fo=missing_length_spec, target_protocol="http")

    with pytest.raises(ValueError):
        missing_offset_spec = {
            "version": 1,
            "templates": {"u": "server.domain/path"},
            "gen": [
                {
                    "key": "gen_key{{i}}",
                    "url": "http://{{u}}_{{i}}",
                    "length": "1000",
                    "dimensions": {"i": {"stop": 2}},
                },
            ],
        }
        fsspec.filesystem("reference", fo=missing_offset_spec, target_protocol="http")

    url_only_gen_spec = {
        "version": 1,
        "templates": {"u": "server.domain/path"},
        "gen": [
            {
                "key": "gen_key{{i}}",
                "url": "http://{{u}}_{{i}}",
                "dimensions": {"i": {"stop": 2}},
            },
        ],
    }

    fs = fsspec.filesystem("reference", fo=url_only_gen_spec, target_protocol="http")
    assert fs.references == {
        "gen_key0": ["http://server.domain/path_0"],
        "gen_key1": ["http://server.domain/path_1"],
    }


def test_empty():
    pytest.importorskip("jinja2")
    fs = fsspec.filesystem("reference", fo={"version": 1}, target_protocol="http")
    assert fs.references == {}


def test_get_sync(tmpdir):
    localfs = LocalFileSystem()

    real = tmpdir / "file"
    real.write_binary(b"0123456789")

    refs = {"a": b"data", "b": (str(real), 0, 5), "c/d": (str(real), 1, 6)}
    fs = fsspec.filesystem("reference", fo=refs, fs=localfs)

    fs.get("a", str(tmpdir / "a"))
    assert (tmpdir / "a").read_binary() == b"data"
    fs.get("b", str(tmpdir / "b"))
    assert (tmpdir / "b").read_binary() == b"01234"
    fs.get("c/d", str(tmpdir / "d"))
    assert (tmpdir / "d").read_binary() == b"123456"
    fs.get("c", str(tmpdir / "c"), recursive=True)
    assert (tmpdir / "c").isdir()
    assert (tmpdir / "c" / "d").read_binary() == b"123456"


def test_multi_fs_provided(m, tmpdir):
    localfs = LocalFileSystem()

    real = tmpdir / "file"
    real.write_binary(b"0123456789")

    m.pipe("afile", b"hello")

    # local URLs are file:// by default
    refs = {
        "a": b"data",
        "b": (f"file://{real}", 0, 5),
        "c/d": (f"file://{real}", 1, 6),
        "c/e": ["memory://afile"],
    }

    fs = fsspec.filesystem("reference", fo=refs, fs={"file": localfs, "memory": m})
    assert fs.cat("c/e") == b"hello"
    assert fs.cat(["c/e", "a", "b"]) == {
        "a": b"data",
        "b": b"01234",
        "c/e": b"hello",
    }


def test_multi_fs_created(m, tmpdir):
    real = tmpdir / "file"
    real.write_binary(b"0123456789")

    m.pipe("afile", b"hello")

    # local URLs are file:// by default
    refs = {
        "a": b"data",
        "b": (f"file://{real}", 0, 5),
        "c/d": (f"file://{real}", 1, 6),
        "c/e": ["memory://afile"],
    }

    fs = fsspec.filesystem("reference", fo=refs, fs={"file": {}, "memory": {}})
    assert fs.cat("c/e") == b"hello"
    assert fs.cat(["c/e", "a", "b"]) == {
        "a": b"data",
        "b": b"01234",
        "c/e": b"hello",
    }


def test_missing_nonasync(m):
    zarr = pytest.importorskip("zarr")
    zarray = {
        "chunks": [1],
        "compressor": None,
        "dtype": "<f8",
        "fill_value": "NaN",
        "filters": [],
        "order": "C",
        "shape": [10],
        "zarr_format": 2,
    }
    refs = {".zarray": json.dumps(zarray)}

    a = zarr.open_array(
        "reference://", storage_options={"fo": refs, "remote_protocol": "memory"}
    )
    assert str(a[0]) == "nan"


def test_fss_has_defaults(m):
    fs = fsspec.filesystem("reference", fo={})
    assert None in fs.fss

    fs = fsspec.filesystem("reference", fo={}, remote_protocol="memory")
    assert fs.fss[None].protocol == "memory"
    assert fs.fss["memory"].protocol == "memory"

    fs = fsspec.filesystem("reference", fs=m, fo={})
    # Default behavior here wraps synchronous filesystems to enable the async API
    assert fs.fss[None].sync_fs is m

    fs = fsspec.filesystem("reference", fs={"memory": m}, fo={})
    assert fs.fss["memory"] is m
    assert fs.fss[None].protocol == ("file", "local")

    fs = fsspec.filesystem("reference", fs={None: m}, fo={})
    assert fs.fss[None] is m

    fs = fsspec.filesystem("reference", fo={"key": ["memory://a"]})
    assert fs.fss[None] == fs.fss["memory"]

    fs = fsspec.filesystem("reference", fo={"key": ["memory://a"], "blah": ["path"]})
    assert fs.fss[None] == fs.fss["memory"]


def test_merging(m):
    m.pipe("/a", b"test data")
    other = b"other test data"
    m.pipe("/b", other)
    fs = fsspec.filesystem(
        "reference",
        fo={
            "a": ["memory://a", 1, 1],
            "b": ["memory://a", 2, 1],
            "c": ["memory://b"],
            "d": ["memory://b", 4, 6],
        },
    )
    out = fs.cat(["a", "b", "c", "d"])
    assert out == {"a": b"e", "b": b"s", "c": other, "d": other[4:10]}


def test_cat_file_ranges(m):
    other = b"other test data"
    m.pipe("/b", other)

    fs = fsspec.filesystem(
        "reference",
        fo={
            "c": ["memory://b"],
            "d": ["memory://b", 4, 6],
        },
    )
    assert fs.cat_file("c") == other
    assert fs.cat_file("c", start=1) == other[1:]
    assert fs.cat_file("c", start=-5) == other[-5:]
    assert fs.cat_file("c", 1, -5) == other[1:-5]

    assert fs.cat_file("d") == other[4:10]
    assert fs.cat_file("d", start=1) == other[4:10][1:]
    assert fs.cat_file("d", start=-5) == other[4:10][-5:]
    assert fs.cat_file("d", 1, -3) == other[4:10][1:-3]


@pytest.mark.asyncio
async def test_async_cat_file_ranges():
    fsspec.get_filesystem_class("http").clear_instance_cache()
    fss = fsspec.filesystem("https", asynchronous=True)
    session = await fss.set_session()

    fs = fsspec.filesystem(
        "reference",
        fo={
            "version": 1,
            "refs": {
                "reference_time/0": [
                    "https://noaa-nwm-retro-v2-0-pds.s3.amazonaws.com/full_physics/2017/201704010000.CHRTOUT_DOMAIN1.comp",
                    39783,
                    12,
                ],
            },
        },
        fs={"https": fss},
        remote_protocol="https",
        asynchronous=True,
    )

    assert (
        await fs._cat_file("reference_time/0") == b"x^K0\xa9d\x04\x00\x03\x13\x01\x0f"
    )
    await session.close()


@pytest.mark.parametrize(
    "fo",
    [
        {
            "c": ["memory://b"],
            "d": ["memory://unknown", 4, 6],
        },
        {
            "c": ["memory://b"],
            "d": ["//unknown", 4, 6],
        },
    ],
    ids=["memory protocol", "mixed protocols: memory and unspecified"],
)
def test_cat_missing(m, fo):
    other = b"other test data"
    m.pipe("/b", other)
    fs = fsspec.filesystem(
        "reference",
        fo=fo,
    )
    with pytest.raises(FileNotFoundError):
        fs.cat("notafile")

    with pytest.raises(FileNotFoundError):
        fs.cat(["notone", "nottwo"])

    mapper = fs.get_mapper("")

    with pytest.raises(KeyError):
        mapper["notakey"]

    with pytest.raises(KeyError):
        mapper.getitems(["notone", "nottwo"])

    with pytest.raises(ReferenceNotReachable) as ex:
        fs.cat("d")
    assert ex.value.__cause__
    out = fs.cat("d", on_error="return")
    assert isinstance(out, ReferenceNotReachable)

    with pytest.raises(ReferenceNotReachable) as e:
        mapper["d"]
    assert '"d"' in str(e.value)
    assert "//unknown" in str(e.value)

    with pytest.raises(ReferenceNotReachable):
        mapper.getitems(["c", "d"])

    out = mapper.getitems(["c", "d"], on_error="return")
    assert isinstance(out["d"], ReferenceNotReachable)

    out = fs.cat(["notone", "c", "d"], on_error="return")
    assert isinstance(out["notone"], FileNotFoundError)
    assert out["c"] == other
    assert isinstance(out["d"], ReferenceNotReachable)

    out = mapper.getitems(["c", "d"], on_error="omit")
    assert list(out) == ["c"]


def test_df_single(m):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("fastparquet")
    data = b"data0data1data2"
    m.pipe({"data": data})
    df = pd.DataFrame(
        {
            "path": [None, "memory://data", "memory://data"],
            "offset": [0, 0, 4],
            "size": [0, 0, 4],
            "raw": [b"raw", None, None],
        }
    )
    df.to_parquet("memory://stuff/refs.0.parq")
    m.pipe(
        ".zmetadata",
        b"""{
    "metadata": {
        ".zgroup": {
            "zarr_format": 2
        },
        "stuff/.zarray": {
            "chunks": [1],
            "compressor": null,
            "dtype": "i8",
            "filters": null,
            "shape": [3],
            "zarr_format": 2
        }
    },
    "zarr_consolidated_format": 1,
    "record_size": 10
    }
    """,
    )
    fs = ReferenceFileSystem(fo="memory:///", remote_protocol="memory")
    allfiles = fs.find("")
    assert ".zmetadata" in allfiles
    assert ".zgroup" in allfiles
    assert "stuff/2" in allfiles

    assert fs.cat("stuff/0") == b"raw"
    assert fs.cat("stuff/1") == data
    assert fs.cat("stuff/2") == data[4:8]


def test_df_multi(m):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("fastparquet")
    data = b"data0data1data2"
    m.pipe({"data": data})
    df0 = pd.DataFrame(
        {
            "path": [None, "memory://data", "memory://data"],
            "offset": [0, 0, 4],
            "size": [0, 0, 4],
            "raw": [b"raw1", None, None],
        }
    )
    df0.to_parquet("memory://stuff/refs.0.parq")
    df1 = pd.DataFrame(
        {
            "path": [None, "memory://data", "memory://data"],
            "offset": [0, 0, 2],
            "size": [0, 0, 2],
            "raw": [b"raw2", None, None],
        }
    )
    df1.to_parquet("memory://stuff/refs.1.parq")
    m.pipe(
        ".zmetadata",
        b"""{
    "metadata": {
        ".zgroup": {
            "zarr_format": 2
        },
        "stuff/.zarray": {
            "chunks": [1],
            "compressor": null,
            "dtype": "i8",
            "filters": null,
            "shape": [6],
            "zarr_format": 2
        }
    },
    "zarr_consolidated_format": 1,
    "record_size": 3
    }
    """,
    )
    fs = ReferenceFileSystem(
        fo="memory:///", remote_protocol="memory", skip_instance_cache=True
    )
    allfiles = fs.find("")
    assert ".zmetadata" in allfiles
    assert ".zgroup" in allfiles
    assert "stuff/2" in allfiles
    assert "stuff/4" in allfiles

    assert fs.cat("stuff/0") == b"raw1"
    assert fs.cat("stuff/1") == data
    assert fs.cat("stuff/2") == data[4:8]
    assert fs.cat("stuff/3") == b"raw2"
    assert fs.cat("stuff/4") == data
    assert fs.cat("stuff/5") == data[2:4]


def test_mapping_getitems(m):
    m.pipe({"a": b"A", "b": b"B"})

    refs = {
        "a": ["a"],
        "b": ["b"],
    }
    h = fsspec.filesystem("memory")
    fs = fsspec.filesystem("reference", fo=refs, fs=h)
    mapping = fs.get_mapper("")
    assert mapping.getitems(["b", "a"]) == {"a": b"A", "b": b"B"}


def test_cached(m, tmpdir):
    fn = f"{tmpdir}/ref.json"

    m.pipe({"a": b"A", "b": b"B"})
    m.pipe("ref.json", b"""{"a": ["a"], "b": ["b"]}""")

    fs = fsspec.filesystem(
        "reference",
        fo="simplecache::memory://ref.json",
        fs=m,
        target_options={"cache_storage": str(tmpdir), "same_names": True},
    )
    assert fs.cat("a") == b"A"
    assert os.path.exists(fn)

    # truncate original file to show we are loading from the cached version
    m.pipe("ref.json", b"")
    fs = fsspec.filesystem(
        "reference",
        fo="simplecache::memory://ref.json",
        fs=m,
        target_options={"cache_storage": str(tmpdir), "same_names": True},
        skip_instance_cache=True,
    )
    assert fs.cat("a") == b"A"


@pytest.fixture()
def lazy_refs(m):
    zarr = pytest.importorskip("zarr")
    skip_zarr_2()
    l = LazyReferenceMapper.create("memory://refs.parquet", fs=m)
    g = zarr.open(
        "reference://",
        storage_options={"fo": "memory://refs.parquet", "remote_options": "memory"},
        zarr_format=2,
        mode="w",
    )
    g.create_dataset(name="data", shape=(100,), chunks=(10,), dtype="int64")
    g.store.fs.references.flush()
    return l


def test_append_parquet(lazy_refs, m):
    pytest.importorskip("kerchunk")
    with pytest.raises(KeyError):
        lazy_refs["data/0"]
    lazy_refs["data/0"] = b"data"
    assert lazy_refs["data/0"] == b"data"
    lazy_refs.flush()

    lazy2 = LazyReferenceMapper("memory://refs.parquet", fs=m)
    assert lazy2["data/0"] == b"data"
    with pytest.raises(KeyError):
        lazy_refs["data/1"]
    lazy2["data/1"] = b"Bdata"
    assert lazy2["data/1"] == b"Bdata"
    lazy2.flush()

    lazy2 = LazyReferenceMapper("memory://refs.parquet", fs=m)
    assert lazy2["data/0"] == b"data"
    assert lazy2["data/1"] == b"Bdata"
    lazy2["data/1"] = b"Adata"
    del lazy2["data/0"]
    assert lazy2["data/1"] == b"Adata"
    assert "data/0" not in lazy2
    lazy2.flush()

    lazy2 = LazyReferenceMapper("memory://refs.parquet", fs=m)
    with pytest.raises(KeyError):
        lazy2["data/0"]
    assert lazy2["data/1"] == b"Adata"


def skip_zarr_2():
    import zarr
    from packaging.version import parse

    if parse(zarr.__version__) < parse("3.0"):
        pytest.skip("Zarr 3 required")


@pytest.mark.parametrize("engine", ["fastparquet", "pyarrow"])
def test_deep_parq(m, engine):
    pytest.importorskip("kerchunk")
    zarr = pytest.importorskip("zarr")
    skip_zarr_2()

    lz = fsspec.implementations.reference.LazyReferenceMapper.create(
        "memory://out.parq",
        fs=m,
        engine=engine,
    )
    g = zarr.open_group(
        "reference://",
        mode="w",
        storage_options={"fo": "memory://out.parq", "remote_protocol": "memory"},
        zarr_version=2,
    )

    g2 = g.create_group("instant")
    arr = g2.create_dataset(name="one", shape=(3,), dtype="int64")
    arr[:] = [1, 2, 3]
    g.store.fs.references.flush()
    lz.flush()

    lz = fsspec.implementations.reference.LazyReferenceMapper(
        "memory://out.parq", fs=m, engine=engine
    )
    g = zarr.open_group(
        "reference://",
        storage_options={"fo": "memory://out.parq", "remote_protocol": "memory"},
        zarr_version=2,
    )
    assert g["instant"]["one"][:].tolist() == [1, 2, 3]
    assert sorted(_["name"] for _ in lz.ls("")) == [
        ".zattrs",
        ".zgroup",
        ".zmetadata",
        "instant",
    ]
    assert sorted(_["name"] for _ in lz.ls("instant")) == [
        "instant/.zattrs",
        "instant/.zgroup",
        "instant/one",
    ]

    assert sorted(_["name"] for _ in lz.ls("instant/one")) == [
        "instant/one/.zarray",
        "instant/one/.zattrs",
        "instant/one/0",
    ]


def test_parquet_no_data(m):
    zarr = pytest.importorskip("zarr")
    skip_zarr_2()
    fsspec.implementations.reference.LazyReferenceMapper.create(
        "memory://out.parq", fs=m
    )
    g = zarr.open_group(
        "reference://",
        storage_options={
            "fo": "memory://out.parq",
            "fs": m,
            "remote_protocol": "memory",
        },
        zarr_format=2,
        mode="w",
    )
    arr = g.create_dataset(
        name="one",
        dtype="int32",
        shape=(10,),
        chunks=(5,),
        compressor=None,
        fill_value=1,
    )
    g.store.fs.references.flush()

    assert (arr[:] == 1).all()


def test_parquet_no_references(m):
    zarr = pytest.importorskip("zarr")
    skip_zarr_2()
    lz = fsspec.implementations.reference.LazyReferenceMapper.create(
        "memory://out.parq", fs=m
    )

    g = zarr.open_group(
        "reference://",
        storage_options={
            "fo": "memory://out.parq",
            "fs": m,
            "remote_protocol": "memory",
        },
        zarr_format=2,
        mode="w",
    )
    arr = g.create_dataset(
        name="one",
        dtype="int32",
        shape=(),
        chunks=(),
        compressor=None,
        fill_value=1,
    )
    lz.flush()

    assert arr[...].tolist() == 1  #  scalar, equal to fill value


def _counting_references(refs):
    """A plain references dict that counts how often it is iterated.

    Iterating the references is exactly the full-scan cost we want to avoid
    when browsing a listing more than once, so tests can assert on ``iterations``
    and on the per-lookup counters in ``fs._dir_stats``.
    """
    from collections import UserDict

    class CountingDict(UserDict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            return super().__iter__()

    return CountingDict(refs)


def test_ls_is_not_quadratic_in_siblings():
    # 4000 siblings: a listing, then per-file ``ls``/``info`` must not rescan
    # the collection each time (previously ~32M name comparisons).
    refs = {f"collection/file{i:04d}": b"x" for i in range(4000)}
    counting = _counting_references(refs)
    fs = fsspec.filesystem("reference", fo=counting, skip_instance_cache=True)

    assert fs.ls("collection", detail=False)[0] == "collection/file0000"
    builds = fs._dir_stats["builds"]
    iterations = counting.iterations
    assert builds == 1
    assert fs._dir_stats["keys_indexed"] == 4000

    # browsing each result individually reuses the index
    for name in ("collection/file0000", "collection/file2000"):
        for _ in range(50):
            assert fs.ls(name, detail=False) == [name]
            assert fs.info(name)["size"] == 1
    assert counting.iterations == iterations  # no full scan per lookup
    assert fs._dir_stats["builds"] == builds

    before = counting.iterations
    found = fs.find("collection", detail=True)
    assert len(found) == 4000
    assert found["collection/file3999"] == {
        "name": "collection/file3999",
        "type": "file",
        "size": 1,
    }
    # enumerating the results costs at most one pass, not one per result
    assert counting.iterations - before <= 1
    after = counting.iterations
    assert fs.find("collection", detail=True) == found
    assert counting.iterations - after <= 1
    assert fs._dir_stats["builds"] == builds


def test_scanning_counts_do_not_grow_with_rescans():
    """Reproduce the reported slowdown with explicit work counters.

    Browsing a 4000-sibling collection used to compare every listed name
    against every sibling (~32M comparisons); here every lookup is counted
    and must be answered from the index instead of a rescan.
    """
    refs = {"collection/target": b"abc"}
    refs.update({f"collection/file{i:04d}": b"x" for i in range(4000)})
    counting = _counting_references(refs)
    fs = fsspec.filesystem("reference", fo=counting, skip_instance_cache=True)

    first = fs.ls("collection", detail=True)
    assert len(first) == 4001
    assert fs._dir_stats["builds"] == 1
    assert fs._dir_stats["keys_indexed"] == 4001
    builds = fs._dir_stats["builds"]
    iterations = counting.iterations

    # listing every result individually costs one lookup each, no rescan
    all_names = fs.find("collection", detail=False)
    before_find = counting.iterations
    lookups = fs._dir_stats["lookups"]
    for entry in all_names:
        assert fs.ls(entry, detail=False) == [entry]
    assert fs._dir_stats["lookups"] - lookups == len(all_names)
    assert fs._dir_stats["builds"] == builds
    assert counting.iterations - before_find <= 1

    # the 3 -> 11 byte overwrite updates exactly one entry, in place
    updates = fs._dir_stats["file_updates"]
    before_write = counting.iterations
    fs.pipe_file("collection/target", b"0123456789a")
    assert fs._dir_stats["file_updates"] - updates == 1
    assert fs.info("collection/target")["size"] == 11
    assert fs.ls("collection/target", detail=True) == [
        {"name": "collection/target", "type": "file", "size": 11}
    ]
    assert fs.find("collection", detail=True)["collection/target"]["size"] == 11
    assert counting.iterations - before_write <= 1
    assert fs._dir_stats["builds"] == builds


def test_ls_single_directory_does_not_scan_other_directories():
    refs = {"wanted/a": b"1", "wanted/b": b"2"}
    refs.update({f"other{i}/file": b"x" for i in range(500)})
    counting = _counting_references(refs)
    fs = fsspec.filesystem("reference", fo=counting, skip_instance_cache=True)

    assert fs.ls("wanted", detail=False) == ["wanted/a", "wanted/b"]
    iterations = counting.iterations

    # a single small-directory update and re-list must not rescan the rest
    fs.pipe_file("wanted/c", b"333")
    assert fs.ls("wanted", detail=False) == ["wanted/a", "wanted/b", "wanted/c"]
    assert fs.info("wanted/c")["size"] == 3
    fs.rm_file("wanted/b")
    assert fs.ls("wanted", detail=False) == ["wanted/a", "wanted/c"]
    assert counting.iterations == iterations
    # only the intended entries were touched
    assert fs._dir_stats["file_updates"] == 1
    assert fs._dir_stats["file_removals"] == 1


def test_listing_does_not_touch_targets(m):
    # sizes come from the reference set; listing must not fetch target bytes
    m.pipe("data", b"0123456789")
    calls = []
    orig = m.cat_file

    def counting_cat_file(*args, **kwargs):
        calls.append(args)
        return orig(*args, **kwargs)

    m.cat_file = counting_cat_file
    try:
        refs = {
            "a": ["memory://data", 0, 3],
            "b": ["memory://data"],
            "c/d": b"inline",
        }
        fs = fsspec.filesystem("reference", fo=refs, fs=m, skip_instance_cache=True)
        assert fs.ls("", detail=True) == [
            {"name": "a", "type": "file", "size": 3},
            {"name": "b", "type": "file", "size": None},
            {"name": "c", "type": "directory", "size": 0},
        ]
        assert fs.find("", detail=True)["b"]["size"] is None
        assert fs.find("", withdirs=True) == ["a", "b", "c", "c/d"]
        assert calls == []
    finally:
        m.cat_file = orig


def _assert_mutation_visible(fs, path, data, new_size):
    """The same instance must reflect a write in every listing API."""
    assert fs.info(path)["size"] == new_size
    assert fs.isfile(path)
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    assert path in fs.ls(parent, detail=False)
    assert path in fs.find("")
    assert fs.find("", detail=True)[path]["size"] == new_size
    assert fs.isfile(path)


def test_sync_pipe_file_updates_listing(tmpdir):
    fs = fsspec.filesystem(
        "reference", fo={"a": b"xyz", "c/d": b"123"}, skip_instance_cache=True
    )
    assert fs.ls("", detail=False) == ["a", "c"]
    assert fs.info("a")["size"] == 3

    fs.pipe_file("a", b"0123456789a")  # 3 -> 11 bytes, same instance
    _assert_mutation_visible(fs, "a", b"0123456789a", 11)
    assert fs.ls("", detail=True) == [
        {"name": "a", "type": "file", "size": 11},
        {"name": "c", "type": "directory", "size": 0},
    ]

    # adding a/b/c shows ancestors, deleting the last child removes them
    fs.pipe_file("new/deep/f", b"q")
    assert fs.ls("", detail=False) == ["a", "c", "new"]
    assert fs.isdir("new") and fs.isdir("new/deep")
    assert fs.info("new") == {"name": "new", "type": "directory", "size": 0}
    assert fs.find("new") == ["new/deep/f"]
    fs.rm_file("new/deep/f")
    assert not fs.isdir("new/deep")
    assert not fs.isdir("new")
    assert fs.ls("", detail=False) == ["a", "c"]
    assert fs.find("", withdirs=True) == ["a", "c", "c/d"]

    # pruning stops at a directory that still has another child
    fs.rm_file("c/d")
    assert not fs.isdir("c")
    assert fs.ls("", detail=False) == ["a"]


def test_sync_put_file_updates_listing(tmpdir):
    src = str(tmpdir / "src")
    with open(src, "wb") as f:
        f.write(b"hello")

    fs = fsspec.filesystem("reference", fo={}, skip_instance_cache=True)
    assert fs.ls("") == []
    fs.put_file(src, "p/q.txt")
    assert fs.ls("p", detail=False) == ["p/q.txt"]
    assert fs.info("p/q.txt")["size"] == 5
    assert fs.isdir("p")
    assert fs.find("", withdirs=True) == ["p", "p/q.txt"]
    assert fs.cat("p/q.txt") == b"hello"


@pytest.mark.asyncio
async def test_async_mutations_update_listing(tmpdir):
    fs = fsspec.filesystem(
        "reference",
        fo={"a": b"xyz", "c/d": b"123"},
        skip_instance_cache=True,
        asynchronous=True,
    )
    await fs._ls("")
    await fs._pipe_file("a", b"0123456789a")
    _assert_mutation_visible(fs, "a", b"0123456789a", 11)

    await fs._pipe_file("n/e/w", b"q")
    assert await fs._isdir("n") and await fs._isdir("n/e")
    assert await fs._ls("", detail=False) == ["a", "c", "n"]
    assert await fs._find("n") == ["n/e/w"]

    await fs._rm_file("n/e/w")
    assert not await fs._isdir("n")
    assert await fs._ls("", detail=False) == ["a", "c"]

    src = str(tmpdir / "src")
    with open(src, "wb") as f:
        f.write(b"hello")
    await fs._put_file(src, "put/me")
    assert await fs._ls("put", detail=False) == ["put/me"]
    assert (await fs._info("put/me"))["size"] == 5
    assert await fs._find("", withdirs=True) == ["a", "c", "c/d", "put", "put/me"]

    await fs._rm_file("c/d")
    assert not await fs._isdir("c")
    assert await fs._ls("", detail=False) == ["a", "put"]


def test_removing_one_of_file_and_directory_same_name():
    fs = fsspec.filesystem(
        "reference", fo={"a": b"x", "a/b": b"y"}, skip_instance_cache=True
    )
    assert fs.ls("", detail=True) == [
        {"name": "a", "type": "file", "size": 1},
        {"name": "a", "type": "directory", "size": 0},
    ]
    fs.rm_file("a")
    # the directory remains because it still has a child
    assert fs.ls("", detail=True) == [
        {"name": "a", "type": "directory", "size": 0},
    ]
    assert fs.isdir("a")
    fs.rm_file("a/b")
    assert fs.ls("", detail=True) == []
    assert not fs.isdir("a")
