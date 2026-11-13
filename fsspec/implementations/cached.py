from __future__ import annotations

import asyncio
import inspect
import logging
import os
import secrets
import tempfile
import time
import weakref
from collections.abc import Callable
from contextlib import contextmanager, suppress
from hashlib import sha256
from shutil import rmtree
from typing import TYPE_CHECKING, Any, ClassVar

from fsspec import filesystem
from fsspec.asyn import _run_coros_in_chunks
from fsspec.callbacks import DEFAULT_CALLBACK
from fsspec.compression import compr
from fsspec.core import BaseCache, MMapCache
from fsspec.exceptions import BlocksizeMismatchError
from fsspec.implementations.cache_mapper import create_cache_mapper
from fsspec.implementations.cache_metadata import CacheMetadata
from fsspec.implementations.chained import ChainedFileSystem
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
from fsspec.transaction import Transaction
from fsspec.utils import infer_compression

if TYPE_CHECKING:
    from fsspec.implementations.cache_mapper import AbstractCacheMapper

logger = logging.getLogger("fsspec.cached")


def _publish(tmp, lpath, start):
    """Move a finished temporary download to its final cache name."""
    try:
        os.replace(tmp, lpath)
    except PermissionError:
        # Windows: the destination is open elsewhere. A same-size file
        # written since this download began is a concurrent complete copy
        # of the same remote file, and makes this one redundant.
        if not (
            os.path.isfile(lpath)
            and os.path.getsize(lpath) == os.path.getsize(tmp)
            and os.path.getmtime(lpath) >= start
        ):
            raise


@contextmanager
def _tempfile(lpath):
    """Yield a temporary name to download into, then publish it.

    Downloads are written beside their final cache location and only renamed
    into place once complete, so that a reader never sees a partially written
    file under the name that marks a cache entry as usable (#639). Temporary
    files are removed when the download or the rename fails; a stale ``.part``
    file left by a killed process is never treated as a cache entry.
    """
    tmp, start = _make_temp(lpath)
    try:
        yield tmp
        _publish(tmp, lpath, start)
    finally:
        with suppress(OSError):
            os.remove(tmp)


def _make_temp(lpath):
    """Create and remove a placeholder temp name, returning it and its mtime.

    The timestamp is read from the cache filesystem's clock (rather than
    ``time.time()``), so it can be compared with a destination's mtime.
    """
    tmp = os.path.join(os.path.dirname(lpath), f"{secrets.token_hex(8)}.part")
    open(tmp, "wb").close()
    start = os.path.getmtime(tmp)
    os.remove(tmp)
    return tmp, start


def _atomic_get(fs, rpaths, lpaths):
    """Download several files in one batch, publishing only complete ones.

    Every target is written to its own temporary file first. If the backend's
    batch call fails, each temporary file whose length matches the remote
    object is still published; partial and absent downloads are discarded.
    Files that completed before the failure are therefore kept, while nothing
    that only partly arrived becomes visible as a cache entry.

    Returns a dict mapping the remote paths that did not complete to the
    exception raised by the batch.
    """
    tmps = []
    starts = []
    for fn in lpaths:
        tmp, start = _make_temp(fn)
        tmps.append(tmp)
        starts.append(start)
    error = None
    try:
        fs.get(rpaths, tmps)
    except Exception as e:
        error = e
    errors = {}
    for rpath, tmp, fn, start in zip(rpaths, tmps, lpaths, starts):
        complete = error is None
        if error is not None:
            # the batch stopped part-way: keep only files whose full length
            # arrived, so a retry downloads exactly what is still missing
            complete = _download_complete(fs, rpath, tmp)
        if not complete:
            with suppress(OSError):
                os.remove(tmp)
            errors[rpath] = error
            continue
        try:
            _publish(tmp, fn, start)
        except OSError as e:
            errors[rpath] = e
        finally:
            with suppress(OSError):
                os.remove(tmp)
    return errors


def _download_complete(fs, rpath, tmp):
    """Whether a temporary file holds the complete remote object."""
    try:
        size = fs.size(rpath)
        return size is not None and os.path.getsize(tmp) == size
    except Exception:
        return False


class WriteCachedTransaction(Transaction):
    def complete(self, commit=True):
        rpaths = [f.path for f in self.files]
        lpaths = [f.fn for f in self.files]
        if commit:
            self.fs.put(lpaths, rpaths)
        self.files.clear()
        self.fs._intrans = False
        self.fs._transaction = None
        self.fs = None  # break cycle


class CachingFileSystem(ChainedFileSystem):
    """Locally caching filesystem, layer over any other FS

    This class implements chunk-wise local storage of remote files, for quick
    access after the initial download. The files are stored in a given
    directory with hashes of URLs for the filenames. If no directory is given,
    a temporary one is used, which should be cleaned up by the OS after the
    process ends. The files themselves are sparse (as implemented in
    :class:`~fsspec.caching.MMapCache`), so only the data which is accessed
    takes up space.

    Restrictions:

    - the block-size must be the same for each access of a given file, unless
      all blocks of the file have already been read
    - caching can only be applied to file-systems which produce files
      derived from fsspec.spec.AbstractBufferedFile ; LocalFileSystem is also
      allowed, for testing
    """

    protocol: ClassVar[str | tuple[str, ...]] = ("blockcache", "cached")
    _strip_tokenize_options = ("fo",)

    def __init__(
        self,
        target_protocol=None,
        cache_storage="TMP",
        cache_check=10,
        check_files=False,
        expiry_time=604800,
        target_options=None,
        fs=None,
        same_names: bool | None = None,
        compression=None,
        cache_mapper: AbstractCacheMapper | None = None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        target_protocol: str (optional)
            Target filesystem protocol. Provide either this or ``fs``.
        cache_storage: str or list(str)
            Location to store files. If "TMP", this is a temporary directory,
            and will be cleaned up by the OS when this process ends (or later).
            If a list, each location will be tried in the order given, but
            only the last will be considered writable.
        cache_check: int
            Number of seconds between reload of cache metadata
        check_files: bool
            Whether to explicitly see if the UID of the remote file matches
            the stored one before using. Warning: some file systems such as
            HTTP cannot reliably give a unique hash of the contents of some
            path, so be sure to set this option to False.
        expiry_time: int
            The time in seconds after which a local copy is considered useless.
            Set to falsy to prevent expiry. The default is equivalent to one
            week.
        target_options: dict or None
            Passed to the instantiation of the FS, if fs is None.
        fs: filesystem instance
            The target filesystem to run against. Provide this or ``protocol``.
        same_names: bool (optional)
            By default, target URLs are hashed using a ``HashCacheMapper`` so
            that files from different backends with the same basename do not
            conflict. If this argument is ``true``, a ``BasenameCacheMapper``
            is used instead. Other cache mapper options are available by using
            the ``cache_mapper`` keyword argument. Only one of this and
            ``cache_mapper`` should be specified.
        compression: str (optional)
            To decompress on download. Can be 'infer' (guess from the URL name),
            one of the entries in ``fsspec.compression.compr``, or None for no
            decompression.
        cache_mapper: AbstractCacheMapper (optional)
            The object use to map from original filenames to cached filenames.
            Only one of this and ``same_names`` should be specified.
        """
        super().__init__(**kwargs)
        if fs is None and target_protocol is None:
            raise ValueError(
                "Please provide filesystem instance(fs) or target_protocol"
            )
        if not (fs is None) ^ (target_protocol is None):
            raise ValueError(
                "Both filesystems (fs) and target_protocol may not be both given."
            )
        if cache_storage == "TMP":
            tempdir = tempfile.mkdtemp()
            storage = [tempdir]
            weakref.finalize(self, self._remove_tempdir, tempdir)
        else:
            if isinstance(cache_storage, str):
                storage = [cache_storage]
            else:
                storage = cache_storage
        os.makedirs(storage[-1], exist_ok=True)
        self.storage = storage
        self.kwargs = target_options or {}
        self.cache_check = cache_check
        self.check_files = check_files
        self.expiry = expiry_time
        self.compression = compression

        # Size of cache in bytes. If None then the size is unknown and will be
        # recalculated the next time cache_size() is called. On writes to the
        # cache this is reset to None.
        self._cache_size = None

        if same_names is not None and cache_mapper is not None:
            raise ValueError(
                "Cannot specify both same_names and cache_mapper in "
                "CachingFileSystem.__init__"
            )
        if cache_mapper is not None:
            self._mapper = cache_mapper
        else:
            self._mapper = create_cache_mapper(
                same_names if same_names is not None else False
            )

        self.target_protocol = (
            target_protocol
            if isinstance(target_protocol, str)
            else (fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0])
        )
        self._metadata = CacheMetadata(self.storage)
        self.load_cache()
        self.fs = fs if fs is not None else filesystem(target_protocol, **self.kwargs)

        def _strip_protocol(path):
            # acts as a method, since each instance has a difference target
            return self.fs._strip_protocol(type(self)._strip_protocol(path))

        self._strip_protocol: Callable = _strip_protocol

    @staticmethod
    def _remove_tempdir(tempdir):
        try:
            rmtree(tempdir)
        except Exception:
            pass

    def _mkcache(self):
        os.makedirs(self.storage[-1], exist_ok=True)

    def cache_size(self):
        """Return size of cache in bytes.

        If more than one cache directory is in use, only the size of the last
        one (the writable cache directory) is returned.
        """
        if self._cache_size is None:
            cache_dir = self.storage[-1]
            self._cache_size = filesystem("file").du(cache_dir, withdirs=True)
        return self._cache_size

    def load_cache(self):
        """Read set of stored blocks from file"""
        self._metadata.load()
        self._mkcache()
        self.last_cache = time.time()

    def save_cache(self):
        """Save set of stored blocks from file"""
        self._mkcache()
        self._metadata.save()
        self.last_cache = time.time()
        self._cache_size = None

    def _check_cache(self):
        """Reload caches if time elapsed or any disappeared"""
        self._mkcache()
        if not self.cache_check:
            # explicitly told not to bother checking
            return
        timecond = time.time() - self.last_cache > self.cache_check
        existcond = all(os.path.exists(storage) for storage in self.storage)
        if timecond or not existcond:
            self.load_cache()

    def _check_file(self, path):
        """Is path in cache and still valid"""
        path = self._strip_protocol(path)
        self._check_cache()
        return self._metadata.check_file(path, self)

    def clear_cache(self):
        """Remove all files and metadata from the cache

        In the case of multiple cache locations, this clears only the last one,
        which is assumed to be the read/write one.
        """
        rmtree(self.storage[-1])
        self.load_cache()
        self._cache_size = None

    def clear_expired_cache(self, expiry_time=None):
        """Remove all expired files and metadata from the cache

        In the case of multiple cache locations, this clears only the last one,
        which is assumed to be the read/write one.

        Parameters
        ----------
        expiry_time: int
            The time in seconds after which a local copy is considered useless.
            If not defined the default is equivalent to the attribute from the
            file caching instantiation.
        """

        if not expiry_time:
            expiry_time = self.expiry

        self._check_cache()

        expired_files, writable_cache_empty = self._metadata.clear_expired(expiry_time)
        for fn in expired_files:
            if os.path.exists(fn):
                os.remove(fn)

        if writable_cache_empty:
            rmtree(self.storage[-1])
            self.load_cache()

        self._cache_size = None

    def pop_from_cache(self, path):
        """Remove cached version of given file

        Deletes local copy of the given (remote) path. If it is found in a cache
        location which is not the last, it is assumed to be read-only, and
        raises PermissionError
        """
        path = self._strip_protocol(path)
        fn = self._metadata.pop_file(path)
        if fn is not None:
            os.remove(fn)
        self._cache_size = None

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        **kwargs,
    ):
        """Wrap the target _open

        If the whole file exists in the cache, just open it locally and
        return that.

        Otherwise, open the file on the target FS, and make it have a mmap
        cache pointing to the location which we determine, in our cache.
        The ``blocks`` instance is shared, so as the mmap cache instance
        updates, so does the entry in our ``cached_files`` attribute.
        We monkey-patch this file, so that when it closes, we call
        ``close_and_update`` to save the state of the blocks.
        """
        path = self._strip_protocol(path)

        path = self.fs._strip_protocol(path)
        if "r" not in mode:
            return self.fs._open(
                path,
                mode=mode,
                block_size=block_size,
                autocommit=autocommit,
                cache_options=cache_options,
                **kwargs,
            )
        detail = self._check_file(path)
        if detail:
            # file is in cache
            detail, fn = detail
            hash, blocks = detail["fn"], detail["blocks"]
            if blocks is True:
                # stored file is complete
                logger.debug("Opening local copy of %s", path)
                return open(fn, mode)
            # TODO: action where partial file exists in read-only cache
            logger.debug("Opening partially cached copy of %s", path)
        else:
            hash = self._mapper(path)
            fn = os.path.join(self.storage[-1], hash)
            blocks = set()
            detail = {
                "original": path,
                "fn": hash,
                "blocks": blocks,
                "time": time.time(),
                "uid": self.fs.ukey(path),
            }
            self._metadata.update_file(path, detail)
            logger.debug("Creating local sparse file for %s", path)

        # explicitly submitting the size to the open call will avoid extra
        # operations when opening. This is particularly relevant
        # for any file that is read over a network, e.g. S3.
        size = detail.get("size")

        # call target filesystems open
        self._mkcache()
        f = self.fs._open(
            path,
            mode=mode,
            block_size=block_size,
            autocommit=autocommit,
            cache_options=cache_options,
            cache_type="none",
            size=size,
            **kwargs,
        )

        # set size if not already set
        if size is None:
            detail["size"] = f.size
            self._metadata.update_file(path, detail)

        if self.compression:
            comp = (
                infer_compression(path)
                if self.compression == "infer"
                else self.compression
            )
            f = compr[comp](f, mode="rb")
        if "blocksize" in detail:
            if detail["blocksize"] != f.blocksize:
                raise BlocksizeMismatchError(
                    f"Cached file must be reopened with same block"
                    f" size as original (old: {detail['blocksize']},"
                    f" new {f.blocksize})"
                )
        else:
            detail["blocksize"] = f.blocksize

        def _fetch_ranges(ranges):
            return self.fs.cat_ranges(
                [path] * len(ranges),
                [r[0] for r in ranges],
                [r[1] for r in ranges],
                **kwargs,
            )

        multi_fetcher = None if self.compression else _fetch_ranges
        f.cache = MMapCache(
            f.blocksize, f._fetch_range, f.size, fn, blocks, multi_fetcher=multi_fetcher
        )
        close = f.close
        f.close = lambda: self.close_and_update(f, close)
        self.save_cache()
        return f

    def _parent(self, path):
        return self.fs._parent(path)

    def hash_name(self, path: str, *args: Any) -> str:
        # Kept for backward compatibility with downstream libraries.
        # Ignores extra arguments, previously same_name boolean.
        return self._mapper(path)

    def close_and_update(self, f, close):
        """Called when a file is closing, so store the set of blocks"""
        if f.closed:
            return
        path = self._strip_protocol(f.path)
        self._metadata.on_close_cached_file(f, path)
        try:
            logger.debug("going to save")
            self.save_cache()
            logger.debug("saved")
        except OSError:
            logger.debug("Cache saving failed while closing file")
        except NameError:
            logger.debug("Cache save failed due to interpreter shutdown")
        close()
        f.closed = True

    def ls(self, path, detail=True):
        return self.fs.ls(path, detail)

    def __getattribute__(self, item):
        if item in {
            "load_cache",
            "_open",
            "save_cache",
            "close_and_update",
            "__init__",
            "__getattribute__",
            "__reduce__",
            "_make_local_details",
            "_detail_and_fn",
            "_cached_now",
            "_cache_paths",
            "_cache_paths_async",
            "_record_download",
            "_record_download_async",
            "_download_one",
            "_download_into",
            "_copy_decoded",
            "_decompress_file",
            "_download_missing",
            "_download_missing_async",
            "_atomic_get_async",
            "_download_async",
            "_check_file_async",
            "_ukey_async",
            "_local_paths_for_ranges",
            "open",
            "cat",
            "cat_file",
            "_cat_file",
            "cat_ranges",
            "_cat_ranges",
            "get",
            "read_block",
            "tail",
            "head",
            "info",
            "ls",
            "exists",
            "isfile",
            "isdir",
            "_check_file",
            "_check_cache",
            "_mkcache",
            "clear_cache",
            "clear_expired_cache",
            "pop_from_cache",
            "local_file",
            "_paths_from_path",
            "get_mapper",
            "open_many",
            "commit_many",
            "hash_name",
            "__hash__",
            "__eq__",
            "to_json",
            "to_dict",
            "cache_size",
            "pipe_file",
            "pipe",
            "start_transaction",
            "end_transaction",
        }:
            # all the methods defined in this class. Note `open` here, since
            # it calls `_open`, but is actually in superclass
            if hasattr(type(self), item):
                return lambda *args, **kw: getattr(type(self), item).__get__(self)(
                    *args, **kw
                )
            # method is only listed for the base class; fall through to the
            # wrapped filesystem, just like any other attribute
        if item in ["__reduce_ex__"]:
            raise AttributeError
        if item in ["transaction"]:
            # property
            return type(self).transaction.__get__(self)
        if item in {"_cache", "transaction_type", "protocol"}:
            # class attributes
            return getattr(type(self), item)
        if item == "__class__":
            return type(self)
        d = object.__getattribute__(self, "__dict__")
        fs = d.get("fs", None)  # fs is not immediately defined
        if item in d:
            return d[item]
        elif fs is not None:
            if item in fs.__dict__:
                # attribute of instance
                return fs.__dict__[item]
            # attributed belonging to the target filesystem
            cls = type(fs)
            m = getattr(cls, item)
            if (inspect.isfunction(m) or inspect.isdatadescriptor(m)) and (
                not hasattr(m, "__self__") or m.__self__ is None
            ):
                # instance method
                return m.__get__(fs, cls)
            return m  # class method or attribute
        else:
            # attributes of the superclass, while target is being set up
            return super().__getattribute__(item)

    def __eq__(self, other):
        """Test for equality."""
        if self is other:
            return True
        if not isinstance(other, type(self)):
            return False
        return (
            self.storage == other.storage
            and self.kwargs == other.kwargs
            and self.cache_check == other.cache_check
            and self.check_files == other.check_files
            and self.expiry == other.expiry
            and self.compression == other.compression
            and self._mapper == other._mapper
            and self.target_protocol == other.target_protocol
        )

    def __hash__(self):
        """Calculate hash."""
        return (
            hash(tuple(self.storage))
            ^ hash(str(self.kwargs))
            ^ hash(self.cache_check)
            ^ hash(self.check_files)
            ^ hash(self.expiry)
            ^ hash(self.compression)
            ^ hash(self._mapper)
            ^ hash(self.target_protocol)
        )


class WholeFileCacheFileSystem(CachingFileSystem):
    """Caches whole remote files on first access

    This class is intended as a layer over any other file system, and
    will make a local copy of each file accessed, so that all subsequent
    reads are local. This is similar to ``CachingFileSystem``, but without
    the block-wise functionality and so can work even when sparse files
    are not allowed. See its docstring for definition of the init
    arguments.

    The class still needs access to the remote store for listing files,
    and may refresh cached files.
    """

    protocol = "filecache"
    local_file = True

    def _cache_paths(self, paths):
        """Local cache file for each of ``paths``, and which need downloading.

        This is the single place where whole-file reads decide what is already
        available. A valid local copy is reused; a missing (or stale) one is
        listed for a batched download from the backend. Duplicate paths share
        one cache file and are downloaded once.

        Returns ``(local_paths, remote_paths, local_paths_to_download)``, where
        ``local_paths`` has one entry per input path, in order.
        """
        local = {}
        lpaths = []
        rpaths = []
        download = []
        for p in paths:
            p = self._strip_protocol(p)
            if p not in local:
                cached = self._detail_and_fn(p)
                if not cached or cached[0]["blocks"] is not True:
                    # missing, stale, or a sparse blockcache entry in this
                    # directory: re-download the whole file
                    fn = os.path.join(self.storage[-1], self._mapper(p))
                    if fn not in download:
                        download.append(fn)
                        rpaths.append(p)
                else:
                    fn = cached[1]
                local[p] = fn
            lpaths.append(local[p])
        return lpaths, rpaths, download

    def _record_download(self, path, uid=None):
        """Record a completed whole-file download in the cache metadata."""
        self._make_local_details(path, uid=uid)

    def _local_paths_for_ranges(self, paths):
        """Local cache files for ``paths``, downloading missing ones.

        A path repeated in ``paths`` maps to a single cache file, downloaded
        once. On failure, complete sibling downloads are kept and the backend
        error is raised; ``pathlib``/``zarr``-style callers see that rather
        than a local missing-file error.
        """
        return self._download_missing(paths)

    async def _record_download_async(self, path):
        """Async variant of ``_record_download``, using an async-safe ukey."""
        self._record_download(path, uid=await self._ukey_async(path))

    def _download_into(self, path, fn, **kwargs):
        """Write a complete copy of ``path`` to cache file ``fn``."""
        kwargs.pop("mode", None)
        if self.compression:
            with (
                _tempfile(fn) as tmp,
                self.fs._open(path, mode="rb", **kwargs) as f,
                open(tmp, "wb") as f2,
            ):
                if isinstance(f, AbstractBufferedFile):
                    # want no type of caching if just downloading whole thing
                    f.cache = BaseCache(0, f.cache.fetcher, f.size)
                self._copy_decoded(path, f, f2)
        else:
            with _tempfile(fn) as tmp:
                self.fs.get_file(path, tmp)

    def _copy_decoded(self, path, f, out):
        """Copy an already-opened remote stream into ``out``, decompressing."""
        comp = (
            infer_compression(path) if self.compression == "infer" else self.compression
        )
        f = compr[comp](f, mode="rb")
        data = True
        while data:
            block = getattr(f, "blocksize", 5 * 2**20)
            data = f.read(block)
            out.write(data)

    def _decompress_file(self, path, source, destination):
        """Decompress the local file ``source`` into ``destination``."""
        with open(source, "rb") as f, open(destination, "wb") as out:
            self._copy_decoded(path, f, out)

    def _detail_and_fn(self, path):
        """Normalised ``(detail, local_path)`` for a cached path, else ``None``.

        The whole-file caches differ in what ``_check_file`` returns: filecache
        gives ``(detail, fn)``, simplecache gives just ``fn``.
        """
        hit = self._check_file(path)
        if not hit:
            return None
        if isinstance(hit, tuple):
            return hit
        return {"original": path, "blocks": True}, hit

    def _download_one(self, path, **kwargs):
        """Fetch one whole file and return its local cache path.

        The download is written to a temporary file and only published under
        its final cache name when complete; only then is it recorded, so a
        failed refresh leaves any previous copy and its record untouched.
        """
        fn = os.path.join(self.storage[-1], self._mapper(path))
        self._mkcache()
        self._download_into(path, fn, **kwargs)
        self._cache_size = None
        self._record_download(path)
        self.save_cache()
        return fn

    def _download_missing(self, paths, **kwargs):
        """Fetch the paths that are not cached, as one batch.

        Returns the local cache file for each input path. Each download is
        published under its final cache name only when complete, and recorded
        only when the whole batch has been attempted. A failure keeps the
        files that did complete (and their records), discards partial copies,
        and then raises the backend error; a later read retries only what is
        still missing.
        """
        lpaths, rpaths, download = self._cache_paths(paths)
        if not download:
            return lpaths
        if self.compression:
            # a decompressing copy must go through the target's own open()
            # stream, so it cannot use the backend's batched get
            error = None
            for path in rpaths:
                try:
                    self._download_one(path, **kwargs)
                except Exception as e:
                    error = error or e
            if error is not None:
                raise error
            return self._cache_paths(paths)[0]
        errors = _atomic_get(self.fs, rpaths, download)
        for path in rpaths:
            if path in errors:
                continue
            try:
                self._record_download(path)
            except Exception as e:
                # the copy is complete and usable, but without a record a
                # later check will simply download it again
                errors[path] = e
        self.save_cache()
        if errors:
            raise next(iter(errors.values()))
        return lpaths

    async def _check_file_async(self, path):
        """Async-safe validity check for a cache entry."""
        path = self._strip_protocol(path)
        self._check_cache()
        detail = self._metadata.check_file(path, None)
        if not detail:
            return False
        detail, fn = detail
        if detail["blocks"] is not True:
            # a partial blockcache entry sharing this directory
            return False
        if self.check_files and detail["uid"] != await self._ukey_async(path):
            return False
        if self.expiry and time.time() - detail["time"] > self.expiry:
            return False
        return fn

    async def _ukey_async(self, path):
        """Compute the target's unique key without blocking the event loop."""
        if type(self.fs).ukey is AbstractFileSystem.ukey:
            return sha256(str(await self.fs._info(path)).encode()).hexdigest()
        return await asyncio.to_thread(self.fs.ukey, path)

    async def _cache_paths_async(self, paths):
        """Async equivalent of ``_cache_paths`` for batch async reads."""
        paths = [self._strip_protocol(p) for p in paths]
        local = {}
        lpaths = []
        need = {}
        for p in paths:
            if p not in local:
                local[p] = await self._check_file_async(p) or need.setdefault(
                    p, os.path.join(self.storage[-1], self._mapper(p))
                )
            lpaths.append(local[p])
        return lpaths, need

    async def _download_missing_async(self, paths, **kwargs):
        """Async batch download of the paths that are not cached.

        Mirrors ``_download_missing``: the backend is asked for all missing
        files at once, each is published only when complete, and complete
        siblings survive a failure of another file in the batch.
        """
        lpaths, need = await self._cache_paths_async(paths)
        if not need:
            return lpaths
        if self.compression:
            # decompressing copies need the target's own open() stream
            results = await _run_coros_in_chunks(
                [self._download_async(p, fn, **kwargs) for p, fn in need.items()],
                batch_size=getattr(self.fs, "batch_size", None),
                return_exceptions=True,
            )
            errors = {p: e for p, e in zip(need, results) if isinstance(e, Exception)}
        else:
            errors = await self._atomic_get_async(dict(need), **kwargs)
        self.save_cache()
        if errors:
            raise next(iter(errors.values()))
        return lpaths

    async def _atomic_get_async(self, need, **kwargs):
        """Batched async download into per-file temporary names."""
        rpaths = list(need)
        fns = list(need.values())
        tmps = []
        starts = []
        for fn in fns:
            tmp, start = await asyncio.to_thread(_make_temp, fn)
            tmps.append(tmp)
            starts.append(start)
        error = None
        try:
            await self.fs._get(rpaths, tmps, **kwargs)
        except Exception as e:
            error = e
        errors = {}
        sizes = [None] * len(rpaths)
        if error is not None:
            # ask about each remote object separately: a single missing path
            # must not hide the files that did complete in the batch
            for i, rpath in enumerate(rpaths):
                try:
                    sizes[i] = await self.fs._size(rpath)
                except Exception:
                    pass
        for rpath, tmp, fn, start, size in zip(rpaths, tmps, fns, starts, sizes):
            complete = error is None
            if error is not None:
                complete = size is not None and (
                    await asyncio.to_thread(os.path.getsize, tmp) == size
                )
            try:
                if not complete:
                    errors[rpath] = error
                    continue
                await asyncio.to_thread(_publish, tmp, fn, start)
                await self._record_download_async(rpath)
            except Exception as e:
                errors[rpath] = e
            finally:
                with suppress(OSError):
                    await asyncio.to_thread(os.remove, tmp)
        return errors

    async def _download_async(self, path, fn, **kwargs):
        """Async single-file download, published and recorded when complete."""
        self._mkcache()
        if self.compression:
            raw = os.path.join(os.path.dirname(fn), f"{secrets.token_hex(8)}.part")
            try:
                await self.fs._get_file(path, raw, **kwargs)
                with _tempfile(fn) as tmp:
                    await asyncio.to_thread(self._decompress_file, path, raw, tmp)
            finally:
                with suppress(OSError):
                    os.remove(raw)
        else:
            with _tempfile(fn) as tmp:
                await self.fs._get_file(path, tmp, **kwargs)
        self._cache_size = None
        await self._record_download_async(path)
        self.save_cache()
        return fn

    async def _cat_file(self, path, start=None, end=None, **kwargs):
        logger.debug("async cat_file %s", path)
        path = self._strip_protocol(path)
        fn = await self._check_file_async(path)
        if not fn:
            fn = os.path.join(self.storage[-1], self._mapper(path))
            await self._download_async(path, fn, **kwargs)
        with open(fn, "rb") as f:  # noqa ASYNC230
            if start:
                f.seek(start)
            size = -1 if end is None else end - f.tell()
            return f.read(size)

    async def _cat_ranges(
        self, paths, starts, ends, max_gap=None, on_error="return", **kwargs
    ):
        logger.debug("async cat ranges %s", paths)
        paths = [self._strip_protocol(p) for p in paths]
        errors = {}
        try:
            lpaths = await self._download_missing_async(paths)
        except Exception as e:
            if on_error == "raise":
                raise
            # complete siblings from the same batch are still readable; the
            # backend error is reported for the rest, not a local-miss error
            lpaths, need = await self._cache_paths_async(paths)
            errors = {}
            for i, p in enumerate(paths):
                if p not in need:
                    continue
                try:
                    if not await self._check_file_async(p):
                        errors[i] = e
                except Exception:
                    errors[i] = e
        readable = [i for i, lp in enumerate(lpaths) if i not in errors]
        results = LocalFileSystem().cat_ranges(
            [lpaths[i] for i in readable],
            [starts[i] for i in readable],
            [ends[i] for i in readable],
            max_gap=max_gap,
            on_error="return",
            **kwargs,
        )
        out = [None] * len(paths)
        for i, value in zip(readable, results):
            out[i] = value
        for i, e in errors.items():
            out[i] = e
        if on_error == "return":
            return out
        ex = next((o for o in out if isinstance(o, Exception)), None)
        if ex is not None:
            raise ex
        return out

    def open_many(self, open_files, **kwargs):
        paths = [of.path for of in open_files]
        if "r" in open_files.mode:
            self._mkcache()
        else:
            return [
                LocalTempFile(
                    self.fs,
                    path,
                    mode=open_files.mode,
                    fn=os.path.join(self.storage[-1], self._mapper(path)),
                    **kwargs,
                )
                for path in paths
            ]

        if self.compression:
            raise NotImplementedError
        return [open(fn, mode=open_files.mode) for fn in self._download_missing(paths)]

    def commit_many(self, open_files):
        self.fs.put([f.fn for f in open_files], [f.path for f in open_files])
        [f.close() for f in open_files]
        for f in open_files:
            # in case autocommit is off, and so close did not already delete
            try:
                os.remove(f.name)
            except FileNotFoundError:
                pass
        self._cache_size = None

    def _make_local_details(self, path, uid=None):
        hash = self._mapper(path)
        fn = os.path.join(self.storage[-1], hash)
        detail = {
            "original": path,
            "fn": hash,
            "blocks": True,
            "time": time.time(),
            "uid": self.fs.ukey(path) if uid is None else uid,
        }
        self._metadata.update_file(path, detail)
        logger.debug("Copying %s to local cache", path)
        return fn

    def _cached_now(self, paths):
        """Which of ``paths`` are safe to read from cache right now.

        Used after a batch download failed, to tell the files that completed
        from those that did not. A check that itself errors is treated as a
        miss, so that the original download error is the one reported.
        """
        usable = set()
        for i, p in enumerate(paths):
            try:
                cached = self._detail_and_fn(p)
            except Exception:
                continue
            if cached and cached[0]["blocks"] is True:
                usable.add(i)
        return usable

    def cat(
        self,
        path,
        recursive=False,
        on_error="raise",
        callback=DEFAULT_CALLBACK,
        **kwargs,
    ):
        paths = self.expand_path(
            path, recursive=recursive, maxdepth=kwargs.get("maxdepth")
        )
        out = {}
        errors = {}
        try:
            lpaths = self._download_missing(paths)
        except Exception as e:
            if on_error == "raise":
                raise
            # The batch kept whatever completed usable; read those, and report
            # the backend error for the rest rather than a local-miss error.
            lpaths = self._cache_paths(paths)[0]
            usable = self._cached_now(paths)
            errors = {i: e for i in range(len(paths)) if i not in usable}

        callback.set_size(len(paths) - len(errors))
        for i, (p, lpath) in enumerate(zip(paths, lpaths)):
            if i in errors:
                if on_error == "return":
                    out[p] = errors[i]
            else:
                with open(lpath, "rb") as f:
                    out[p] = f.read()
            callback.relative_update(1)
        if isinstance(path, str) and len(paths) == 1 and recursive is False:
            if paths[0] in out:
                out = out[paths[0]]
        return out

    def _open(self, path, mode="rb", **kwargs):
        path = self._strip_protocol(path)
        if "r" not in mode:
            hash = self._mapper(path)
            fn = os.path.join(self.storage[-1], hash)
            user_specified_kwargs = {
                k: v
                for k, v in kwargs.items()
                # those kwargs were added by open(), we don't want them
                if k not in ["autocommit", "block_size", "cache_options"]
            }
            return LocalTempFile(self, path, mode=mode, fn=fn, **user_specified_kwargs)
        cached = self._detail_and_fn(path)
        if cached:
            detail, fn = cached
            if detail["blocks"] is True:
                logger.debug("Opening local copy of %s", path)

                # In order to support downstream filesystems to be able to
                # infer the compression from the original filename, like
                # the `TarFileSystem`, let's extend the `io.BufferedReader`
                # fileobject protocol by adding a dedicated attribute
                # `original`.
                f = open(fn, mode)
                f.original = detail.get("original")
                return f
            else:
                raise ValueError(
                    f"Attempt to open partially cached file {path}"
                    f" as a wholly cached file"
                )
        kwargs["mode"] = mode

        # call target filesystems open
        self._download_one(path, **kwargs)
        return self._open(path, mode)


class SimpleCacheFileSystem(WholeFileCacheFileSystem):
    """Caches whole remote files on first access

    This class is intended as a layer over any other file system, and
    will make a local copy of each file accessed, so that all subsequent
    reads are local. This implementation only copies whole files, and
    does not keep any metadata about the download time or file details.
    It is therefore safer to use in multi-threaded/concurrent situations.

    This is the only of the caching filesystems that supports write: you will
    be given a real local open file, and upon close and commit, it will be
    uploaded to the target filesystem; the writability or the target URL is
    not checked until that time.

    """

    protocol = "simplecache"
    local_file = True
    transaction_type = WriteCachedTransaction

    def __init__(self, **kwargs):
        kw = kwargs.copy()
        for key in ["cache_check", "expiry_time", "check_files"]:
            kw[key] = False
        super().__init__(**kw)
        for storage in self.storage:
            if not os.path.exists(storage):
                os.makedirs(storage, exist_ok=True)

    def _check_file(self, path):
        self._check_cache()
        sha = self._mapper(path)
        for storage in self.storage:
            fn = os.path.join(storage, sha)
            if os.path.exists(fn):
                return fn

    def save_cache(self):
        pass

    def load_cache(self):
        pass

    def _record_download(self, path, uid=None):
        # simplecache keeps no metadata; the local file itself is the record
        pass

    async def _record_download_async(self, path):
        # simplecache keeps no metadata; the local file itself is the record
        pass

    async def _check_file_async(self, path):
        # existence-based, so the sync check performs no remote I/O
        return self._check_file(path)

    def pipe_file(self, path, value=None, **kwargs):
        if self._intrans:
            with self.open(path, "wb") as f:
                f.write(value)
        else:
            super().pipe_file(path, value)

    def ls(self, path, detail=True, **kwargs):
        path = self._strip_protocol(path)
        details = []
        try:
            details = self.fs.ls(
                path, detail=True, **kwargs
            ).copy()  # don't edit original!
        except FileNotFoundError as e:
            ex = e
        else:
            ex = None
        if self._intrans:
            path1 = path.rstrip("/") + "/"
            for f in self.transaction.files:
                if f.path == path:
                    details.append(
                        {"name": path, "size": f.size or f.tell(), "type": "file"}
                    )
                elif f.path.startswith(path1):
                    if f.path.count("/") == path1.count("/"):
                        details.append(
                            {"name": f.path, "size": f.size or f.tell(), "type": "file"}
                        )
                    else:
                        dname = "/".join(f.path.split("/")[: path1.count("/") + 1])
                        details.append({"name": dname, "size": 0, "type": "directory"})
        if ex is not None and not details:
            raise ex
        if detail:
            return details
        return sorted(_["name"] for _ in details)

    def info(self, path, **kwargs):
        path = self._strip_protocol(path)
        if self._intrans:
            f = [_ for _ in self.transaction.files if _.path == path]
            if f:
                size = os.path.getsize(f[0].fn) if f[0].closed else f[0].tell()
                return {"name": path, "size": size, "type": "file"}
            f = any(_.path.startswith(path + "/") for _ in self.transaction.files)
            if f:
                return {"name": path, "size": 0, "type": "directory"}
        return self.fs.info(path, **kwargs)

    def pipe(self, path, value=None, **kwargs):
        if isinstance(path, str):
            self.pipe_file(self._strip_protocol(path), value, **kwargs)
        elif isinstance(path, dict):
            for k, v in path.items():
                self.pipe_file(self._strip_protocol(k), v, **kwargs)
        else:
            raise ValueError("path must be str or dict")

    def cat_ranges(
        self, paths, starts, ends, max_gap=None, on_error="return", **kwargs
    ):
        logger.debug("cat ranges %s", paths)
        errors = {}
        try:
            lpaths = self._local_paths_for_ranges(paths)
        except Exception as e:
            if on_error == "raise":
                raise
            # complete siblings from the same batch remain readable; the
            # backend error is reported for the paths that are still missing
            lpaths = self._cache_paths(paths)[0]
            usable = self._cached_now(paths)
            errors = {i: e for i in range(len(paths)) if i not in usable}

        readable = [i for i, lp in enumerate(lpaths) if i not in errors]
        results = LocalFileSystem().cat_ranges(
            [lpaths[i] for i in readable],
            [starts[i] for i in readable],
            [ends[i] for i in readable],
            max_gap=max_gap,
            on_error="return",
            **kwargs,
        )
        out = [None] * len(paths)
        for i, value in zip(readable, results):
            out[i] = value
        for i, e in errors.items():
            out[i] = e
        if on_error == "return":
            return out
        ex = next((o for o in out if isinstance(o, Exception)), None)
        if ex is not None:
            raise ex
        return out

    def _open(self, path, mode="rb", **kwargs):
        path = self._strip_protocol(path)
        if "r" not in mode:
            sha = self._mapper(path)
            fn = os.path.join(self.storage[-1], sha)
            user_specified_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in ["autocommit", "block_size", "cache_options"]
            }  # those were added by open()
            return LocalTempFile(
                self,
                path,
                mode=mode,
                autocommit=not self._intrans,
                fn=fn,
                **user_specified_kwargs,
            )
        # reads, including compressed ones, use the shared whole-file flow
        return super()._open(path, mode, **kwargs)


class LocalTempFile:
    """A temporary local file, which will be uploaded on commit"""

    def __init__(self, fs, path, fn, mode="wb", autocommit=True, seek=0, **kwargs):
        self.fn = fn
        self.fh = open(fn, mode)
        self.mode = mode
        if seek:
            self.fh.seek(seek)
        self.path = path
        self.size = None
        self.fs = fs
        self.closed = False
        self.autocommit = autocommit
        self.kwargs = kwargs

    def __reduce__(self):
        # always open in r+b to allow continuing writing at a location
        return (
            LocalTempFile,
            (self.fs, self.path, self.fn, "r+b", self.autocommit, self.tell()),
        )

    def __enter__(self):
        return self.fh

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        # self.size = self.fh.tell()
        if self.closed:
            return
        self.fh.close()
        self.closed = True
        if self.autocommit:
            self.commit()

    def discard(self):
        self.fh.close()
        os.remove(self.fn)

    def commit(self):
        # calling put() with list arguments avoids path expansion and additional operations
        # like isdir()
        self.fs.put([self.fn], [self.path], **self.kwargs)
        # we do not delete the local copy, it's still in the cache.

    @property
    def name(self):
        return self.fn

    def __repr__(self) -> str:
        return f"LocalTempFile: {self.path}"

    def __getattr__(self, item):
        return getattr(self.fh, item)
