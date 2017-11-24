import os
import six
import lmdb
from . import serialization


class LMDB(object):

    def __init__(self, lmdb_path, readonly=True, readahead=True,
                 map_size=1099511627776 * 2, meminit=False, map_async=True,
                 use_serialization=False, serializer=None, **lmdb_kwargs):
        isdir = os.path.isdir(lmdb_path)
        self._lmdb = lmdb.open(lmdb_path, subdir=isdir, readonly=readonly, readahead=readahead,
                               map_size=map_size, meminit=meminit, map_async=map_async, **lmdb_kwargs)

        if use_serialization:
            if serializer is None:
                serialization_name = self._lmdb.get_value(b"__serialization__")
                serialization_name = serialization_name.decode("ascii")
                self._serializer = serialization.get_serializer_by_name(serialization_name)
            self._serializer = serializer
        else:
            self._serializer = serialization.DummySerializer()

    def begin(self, write=False, buffers=False):
        return self._lmdb.begin(write=write, buffers=buffers)

    def get_item(self, key, deserialize=False):
        if isinstance(key, six.string_types):
            key = key.encode("ascii")
        with self._lmdb.begin() as txn:
            v = txn.get(key)
            if deserialize and v is not None:
                v = self._serializer.loads(v)
            return v

    def put_item(self, key, value, sync=True, serialize=False):
        if isinstance(key, six.string_types):
            key = key.encode("ascii")
        # Open same way as tensorpack
        if serialize:
            value = self._serializer.dumps(value)
        with self._lmdb.begin(write=True) as txn:
            txn.put(key, value)
        if sync:
            self._lmdb.sync()

    def delete_item(self, key):
        if isinstance(key, six.string_types):
            key = key.encode("ascii")
        self._lmdb.begin().delete(key)

    def sync(self):
        self._lmdb.sync()

    def close(self):
        self._lmdb.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._lmdb.close()
        return False
