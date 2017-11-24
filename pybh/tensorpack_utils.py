import os
import threading
import six
import zlib
import bz2
import pprint
import multiprocessing
from contextlib import contextmanager
import numpy as np
import zmq
import uuid
from . import lmdb_utils
import tensorflow as tf
import tensorpack
from tensorpack.dataflow.base import DataFlowReentrantGuard
from .utils import logged_time_measurement
from .progressbar import get_progressbar
from . import log_utils
from .utils import Timer
from . import serialization
from . import thread_utils


logger = log_utils.get_logger("pybh.tensorpack_utils")


class SideEffectDataFlow(tensorpack.dataflow.ProxyDataFlow):

    def __init__(self, df, side_effect_func):
        self._side_effect_func = side_effect_func
        super(SideEffectDataFlow, self).__init__(df)

    def get_data(self):
        for dp in self.ds.get_data():
            self._side_effect_func(dp)
            yield dp


class LogMessageDataFlow(SideEffectDataFlow):

    def __init__(self, df, message_func, logger_=None):
        self._message_func = message_func
        if logger_ is None:
            logger_ = logger
        self._logger = logger_
        super(LogMessageDataFlow, self).__init__(df, self._print_message)

    def _print_message(self, dp):
        self._logger.info(self._message_func(dp))

    def get_data(self):
        for dp in self.ds.get_data():
            self._side_effect_func(dp)
            yield dp


class InspectDataFlow(SideEffectDataFlow):

    def __init__(self, df, prefix="Dataflow", logger_=None):
        self._prefix = prefix
        if logger_ is None:
            logger_ = logger
        self._logger = logger_
        super(InspectDataFlow, self).__init__(df, self._debug_datapoint)

    def _get_message(self, dp):
        msg = "type: {}".format(type(dp).__name__)
        try:
            msg += ", len: {}".format(len(dp))
        except TypeError:
            pass
        if isinstance(dp, list):
            msg += ", ["
            for x in dp:
                msg += "{:s}, ".format(self._get_message(x))
            msg += "]"
        if isinstance(dp, dict):
            msg += ", {"
            for k, x in dp.items():
                msg += "{}: {:s}, ".format(k, self._get_message(x))
            msg += "}"
        return msg

    def _debug_datapoint(self, dp):
        msg = self._get_message(dp)
        self._logger.info("{:s}> {:s}".format(self._prefix, msg))

    def get_data(self):
        for dp in self.ds.get_data():
            self._side_effect_func(dp)
            yield dp


class ThroughputMeasureDataFlow(tensorpack.dataflow.ProxyDataFlow):

    def __init__(self, df, name, interval=10):
        self._name = name
        self._interval = interval
        self._timer = Timer()
        super(ThroughputMeasureDataFlow, self).__init__(df)

    def get_data(self):
        self._timer.restart()
        counter = 0
        for dp in self.ds.get_data():
            counter += 1
            if counter % self._interval == 0:
                logger.info("DataFlow {} running with {} Hz".format(self._name, self._timer.compute_rate(counter)))
            yield dp


# Adapted from tensorpack.dataflow.LMDBData class

class LMDBDataWithMetaData(tensorpack.dataflow.RNGDataFlow):
    """ Read a LMDB database and produce (k,v) raw string pairs.
        Ignore keys starting with __.
        Enable fetching of keys starting with __ as metadata.
    """

    def __init__(self, lmdb_path, shuffle=False, keys=None, random_start_position=False,
                 serializer=None, deserialize_datapoints=False, max_num_samples=None):
        """
        Args:
            lmdb_path (str): a directory or a file.
            serializer (obj): serializer object to load data.
            shuffle (bool): shuffle the keys or not.
            keys (list[str] or str): list of str as the keys, used only when shuffle is True.
                It can also be a format string e.g. ``{:0>8d}`` which will be
                formatted with the indices from 0 to *total_size - 1*.
                If not provided, it will then look in the database for ``__keys__`` which
                :func:`dump_dataflow_to_lmdb` used to store the list of keys.
                If still not found, it will iterate over the database to find
                all the keys.
            random_start_position (bool): after each reset start at a random position within the dataset.
            deserialize_datapoints (bool): deserialize datapoints.
        """
        self._lmdb_path = lmdb_path
        self._shuffle = shuffle
        self._random_start_position = random_start_position
        self._max_num_samples = max_num_samples

        self._cursor = None
        self._open_lmdb()
        self._size = self._txn.stat()['entries']

        self._guard = DataFlowReentrantGuard()

        if serializer is None:
            serialization_name = self._lmdb.get_item(b"__serialization__")
            assert serialization_name is not None, "Serializer was not provided and not serialization metadata in LMDB"
            serialization_name = serialization_name.decode("ascii")
            serializer = serialization.get_serializer_by_name(serialization_name)
        self._serializer = serializer
        self._deserialize_datapoints = deserialize_datapoints

        self._set_keys(keys)
        logger.info("Found {} entries in {}".format(self._size, self._lmdb_path))
        if max_num_samples is not None and max_num_samples > 0:
            self._size = min(self._size, max_num_samples)
            logger.info("Only using {} entries".format(self._size))
            if self.keys is not None:
                self.keys = self.keys[:self._size]

    @property
    def serializer(self):
        return self._serializer

    def _set_keys(self, keys=None):
        def find_keys(txn, size):
            logger.warn("Traversing the database to find keys is slow. Your should specify the keys.")
            keys = []
            with logged_time_measurement(logger, "Loading LMDB keys ...", log_start=True), \
                    get_progressbar(total=size) as pbar:
                for k in txn.cursor():
                    assert k[0] != b'__keys__'
                    if not k.startswith(b"__"):
                        keys.append(k[0])
                        pbar.update()
            return keys

        self.keys = self._txn.get(b'__keys__')
        if self.keys is not None:
            self.keys = self._serializer.loads(self.keys)
            # self._size -= 1     # delete this item
            self._size = len(self.keys)

        if self._shuffle:   # keys are necessary when shuffle is True
            if keys is None:
                if self.keys is None:
                    self.keys = find_keys(self._txn, self._size)
            else:
                # check if key-format like '{:0>8d}' was given
                if isinstance(keys, six.string_types):
                    self.keys = map(lambda x: keys.format(x), list(np.arange(self._size)))
                else:
                    self.keys = keys
            self._size = len(self.keys)

    def _open_lmdb(self):
        self._lmdb = lmdb_utils.LMDB(self._lmdb_path, readonly=True, map_size=1099511627776 * 2,
                                     lock=False, max_readers=100)
        self._txn = self._lmdb.begin()

    def close_lmdb(self):
        self._lmdb.close()

    def reset_state(self):
        self._lmdb.close()
        super(LMDBDataWithMetaData, self).reset_state()
        self._open_lmdb()
        if self._random_start_position:
            start_position = self.rng.randint(len(self.keys))
            logger.info("Resetting LMDB dataflow to random start position: {}".format(start_position))
            old_length = len(self.keys)
            self.keys = self.keys[start_position:] + self.keys[:start_position]
            assert len(self.keys) == old_length
        if self._cursor is not None:
            self._cursor = self._txn.cursor()

    def size(self):
        return self._size

    def get_data(self):
        with self._guard:
            if self.keys is None:
                assert not self._shuffle
                assert not self._random_start_position
                logger.info("Iterating through LMDB data in sequential order")
                self._cursor = self._txn.cursor()
                i = 0
                while self._cursor.next():
                    k, v = self._cursor.item()
                    if not bytes(k).startswith(b"__"):
                        if self._deserialize_datapoints:
                            if v is not None:
                                v = self._serializer.loads(v)
                        yield [k, v]
                        i += 1
                        if self._max_num_samples is not None and self._max_num_samples > 0:
                            if i >= self._max_num_samples:
                                break
            else:
                if self._shuffle:
                    logger.info("Iterating through LMDB data in random order")
                    self.rng.shuffle(self.keys)
                elif self._random_start_position:
                    logger.info("Iterating through LMDB data from random start position")
                else:
                    logger.info("Iterating through LMDB data in sequential order")
                for k in self.keys:
                    v = self._txn.get(k)
                    if v is not None:
                        if self._deserialize_datapoints:
                            if v is not None:
                                v = self._serializer.loads(v)
                    yield [k, v]

    def get_keys(self):
        return self.keys

    def get_item(self, key, deserialize=True):
        if isinstance(key, six.integer_types):
            key = self.keys[key]
        with self._guard:
            v = self._txn.get(key)
            if deserialize:
                if v is not None:
                    v = self._serializer.loads(v)
            return v

    def get_metadata(self, key, deserialize=True):
        with self._guard:
            if isinstance(key, six.string_types):
                key = key.encode("ascii")
            if not key.startswith(b"__"):
                key = b"__%s__" % key
            v = self._txn.get(key)
            if deserialize:
                if v is not None:
                    v = self._serializer.loads(v)
            else:
                # Ensure we return a bytes object even if lmdb is in buffer mode
                v = bytes(v)
            return v

    def has_metadata(self, key):
        return self.get_metadata(key, deserialize=False) is not None


# Adapted from tensorpack.dataflow.FixedSizeData class

class FixedSizeData(tensorpack.dataflow.ProxyDataFlow):
    """ Generate data from another DataFlow, but with a fixed total count.
    """

    def __init__(self, ds, size, keep_state=True):
        """
        Args:
            ds (DataFlow): input dataflow
            size (int): size
            keep_state (bool): keep the iterator state of ``ds``
                between calls to :meth:`get_data()`, so that the
                next call will continue the previous iteration over ``ds``,
                instead of reinitializing an iterator.

        Examples:

        .. code-block:: none

            ds produces: 1, 2, 3, 4, 5; 1, 2, 3, 4, 5; ...
            FixedSizeData(ds, 3, True): 1, 2, 3; 4, 5, 1; 2, 3, 4; ...
            FixedSizeData(ds, 3, False): 1, 2, 3; 1, 2, 3; ...
            FixedSizeData(ds, 6, False): 1, 2, 3, 4, 5, 1; 1, 2, 3, 4, 5, 1;...
        """
        super(FixedSizeData, self).__init__(ds)
        self._size = int(size)
        self.itr = None
        self._guard = DataFlowReentrantGuard()
        self._keep = keep_state

    def set_size(self, new_size):
        self._size = new_size

    def size(self):
        return self._size

    def get_data(self):
        with self._guard:
            if self.itr is None:
                self.itr = self.ds.get_data()
            cnt = 0
            while True:
                try:
                    dp = next(self.itr)
                except StopIteration:
                    self.itr = self.ds.get_data()
                    dp = next(self.itr)

                cnt += 1
                yield dp
                if cnt == self._size:
                    if not self._keep:
                        self.itr = None
                    return


# Adapted from tensorpack.dataflow.dftools.dump_dataflow_to_lmdb method

def dump_dataflow_to_lmdb(df, lmdb_path, serializer=None, serialize_datapoints=False,
                          write_frequency=5000, append=False, reset_df_state=True):
    """
    Dump a Dataflow to a lmdb database, where the keys are indices and values
    are serialized datapoints.
    The output database can be read directly by
    :class:`tensorpack.dataflow.LMDBDataPoint`.
    Args:
        df (DataFlow): the DataFlow to dump.
        lmdb_path (str): output path. Either a directory or a mdb file.
        write_frequency (int): the frequency to write back data to disk.
        append (bool): append data to an existing lmdb database.
        pack_with_msgpack (bool): Pack datapoints with msgpack. Otherwise a single component of ``bytes`` type is assumed.
    """
    if serializer is None:
        serializer = serialization.PickleSerializer()

    assert isinstance(df, tensorpack.dataflow.DataFlow), type(df)
    isdir = os.path.isdir(lmdb_path)
    if isdir:
        db_exists = os.path.isfile(os.path.join(lmdb_path, 'data.mdb'))
    else:
        db_exists = os.path.isfile(lmdb_path)
    if append:
        assert db_exists, "Trying to append to non-existent LMDB database"
    else:
        assert not db_exists, "LMDB database does already exist"
    if reset_df_state:
        df.reset_state()
    lmdb_db = lmdb_utils.LMDB(lmdb_path, readonly=False, map_size=1099511627776 * 2,
                              meminit=False, map_async=True)    # need sync() at the end

    if append:
        keys_dump = lmdb_db.get_item(b'__keys__')
        keys = serializer.loads(keys_dump)
        if keys is None:
            raise RuntimeError("Appending to LMDB database without __keys__ field is not supported.")
        start_idx = max([int(key) for key in keys]) + 1
    else:
        keys = []
        start_idx = 0
    try:
        sz = df.size()
    except NotImplementedError:
        sz = 0
    idx = 0
    with get_progressbar(total=sz) as pbar:
        # lmdb transaction is not exception-safe!
        # although it has a contextmanager interface
        txn = lmdb_db.begin(write=True)
        for idx, dp in enumerate(df.get_data()):
            key = u'{}'.format(start_idx + idx).encode('ascii')
            if serialize_datapoints:
                dp = serializer.dumps(dp)
            txn.put(key, dp)
            keys.append(key)
            pbar.update()
            if (idx + 1) % write_frequency == 0:
                txn.commit()
                txn = lmdb_db.begin(write=True)
        txn.commit()

        with lmdb_db.begin(write=True) as txn:
            txn.put(b'__keys__', serializer.dumps(keys))

        logger.info("Flushing database ...")
        lmdb_db.sync()
    lmdb_db.close()

    return start_idx + idx + 1


class AutoDeserializeData(tensorpack.dataflow.ProxyDataFlow):

    def __init__(self, df, serialization_name=None):
        if isinstance(serialization_name, serialization.Serializer):
            serializer = serialization_name
        else:
            serializer = serialization.get_serializer_by_name(serialization_name)
        super(AutoDeserializeData, self).__init__(DeserializeData(df, serializer))


class SerializeData(tensorpack.dataflow.ProxyDataFlow):

    def __init__(self, df, serializer):
        self._serializer = serializer
        super(SerializeData, self).__init__(df)

    def get_data(self):
        for datapoint in self.ds.get_data():
            yield self._serializer.dumps(datapoint)

    def dumps(self, obj):
        self._serializer.dumps(obj)

    def loads(self, dump):
        self._serializer.loads(dump)


class DeserializeData(tensorpack.dataflow.ProxyDataFlow):

    def __init__(self, df, serializer):
        self._serializer = serializer
        super(DeserializeData, self).__init__(df)

    def get_data(self):
        for datapoint in self.ds.get_data():
            yield self._serializer.loads(datapoint)

    def dumps(self, obj):
        self._serializer.dumps(obj)

    def loads(self, dump):
        self._serializer.loads(dump)


class PickleSerializeData(SerializeData):

    def __init__(self, df):
        super(PickleSerializeData, self).__init__(df, serialization.PickleSerializer())


class PickleDeserializeData(DeserializeData):

    def __init__(self, df):
        super(PickleDeserializeData, self).__init__(df, serialization.PickleSerializer())


class MsgPackSerializeData(SerializeData):

    def __init__(self, df):
        super(MsgPackSerializeData, self).__init__(df, serialization.MsgPackSerializer())


class MsgPackDeserializeData(DeserializeData):

    def __init__(self, df):
        super(MsgPackDeserializeData, self).__init__(df, serialization.MsgPackSerializer())


def dump_compressed_dataflow_to_lmdb(df, lmdb_path, batch_size, serialization_name=None,
                                     compression=None, compression_arg=None, write_frequency=5000,
                                     append=False, reset_df_state=True):
    if append:
        with lmdb_utils.LMDB(lmdb_path, readonly=False) as lmdb_db:
            if serialization_name is None:
                serialization_name = lmdb_db.get_item("__serialization__").decode("ascii")

    if serialization_name is None:
        serialization_name = "pickle"
    serializer = serialization.get_serializer_by_name(serialization_name)

    logger.info("Serializer object for dumping dataflow: {}".format(serializer))

    if append:
        with lmdb_utils.LMDB(lmdb_path, readonly=False) as lmdb_db:
            if batch_size > 0:
                assert serializer.loads(lmdb_db.get_item("__batch_size__")) == batch_size
            else:
                assert lmdb_db.get_item("__batch_size__") is None
            if compression is not None:
                assert serializer.loads(lmdb_db.get_item("__compression__")) == compression
            else:
                assert lmdb_db.get_item("__compression__") is None

    if batch_size > 0:
        batch_df = BatchDictData(df, batch_size, remainder=False)
    else:
        batch_df = df

    serialize_df = SerializeData(batch_df, serializer)

    if compression is None:
        compress_df = batch_df
    elif compression == "lz4":
        compress_df = Lz4CompressData(serialize_df, compression_arg)
    elif compression == "snappy":
        compress_df = SnappyCompressData(serialize_df)
    elif compression == "zlib":
        compress_df = ZlibCompressData(serialize_df, int(compression_arg))
    elif compression == "lz4":
        compress_df = Bz2CompressData(serialize_df, int(compression_arg))
    else:
            raise RuntimeError("Unknown compressiontype:{}".format(compression))
    num_datapoints = dump_dataflow_to_lmdb(compress_df, lmdb_path, serializer, serialize_datapoints=False,
                                           write_frequency=write_frequency, append=append,
                                           reset_df_state=reset_df_state)
    logger.info("serialization: {}".format(serialization))
    logger.info("batch_size: {}".format(batch_size))
    logger.info("num_datapoints: {}".format(num_datapoints))
    with lmdb_utils.LMDB(lmdb_path, readonly=False) as lmdb_db:
        lmdb_db.put_item("__serialization__", serialization_name.encode("ascii"))
        if batch_size > 0:
            lmdb_db.put_item("__batch_size__", serializer.dumps(batch_size))
        lmdb_db.put_item("__total_size__", serializer.dumps(num_datapoints * batch_size))
        if compression is not None:
            lmdb_db.put_item("__compression__", serializer.dumps(compression))


class AutoUnbatchData(tensorpack.dataflow.ProxyDataFlow):

    def __init__(self, df, batch_size, total_size=None):
        if batch_size > 0:
            super(AutoUnbatchData, self).__init__(UnbatchData(df, batch_size, total_size=total_size))
        else:
            super(AutoUnbatchData, self).__init__(df)


class AutoUnbatchDictData(tensorpack.dataflow.ProxyDataFlow):

    def __init__(self, df, batch_size, total_size=None):
        if batch_size > 0:
            super(AutoUnbatchDictData, self).__init__(UnbatchDictData(df, batch_size, total_size=total_size))
        else:
            super(AutoUnbatchDictData, self).__init__(df)


class AutoDecompressData(tensorpack.dataflow.ProxyDataFlow):

    def __init__(self, df, compression):
        if compression is None:
            super(AutoDecompressData, self).__init__(df)
        elif compression == "lz4":
            super(AutoDecompressData, self).__init__(Lz4DecompressData(df))
        elif compression == "snappy":
            super(AutoDecompressData, self).__init__(SnappyDecompressData(df))
        elif compression == "zlib":
            super(AutoDecompressData, self).__init__(ZlibDecompressData(df))
        elif compression == "bz2":
            super(AutoDecompressData, self).__init__(Bz2DecompressData(df))
        else:
            raise RuntimeError("Unknown compressiontype:{}".format(compression))


class Lz4CompressData(tensorpack.dataflow.ProxyDataFlow):
    """
    Compresses a datapoint using LZ4.
    """

    def __init__(self, ds, compression_mode="default"):
        """
        Args:
            ds (DataFlow): Input dataflow.
            compression_mode (str): LZ4 compression mode.
        """
        super(Lz4CompressData, self).__init__(ds)
        if compression_mode is None:
            compression_mode = "default"
        self._compression_mode = compression_mode

    def get_data(self):
        """
        Yields: Compressed datapoint.
        """
        import lz4
        for datapoint in self.ds.get_data():
            compressed_datapoint = lz4.block.compress(datapoint, self._compression_mode)
            yield compressed_datapoint


class Lz4DecompressData(tensorpack.dataflow.ProxyDataFlow):
    """
    Decompressed a datapoint using LZ4.
    """

    def __init__(self, ds):
        """
        Args:
            ds (DataFlow): Input dataflow.
        """
        super(Lz4DecompressData, self).__init__(ds)

    def get_data(self):
        """
        Yields: Decompressed datapoint.
        """
        import lz4
        for datapoint in self.ds.get_data():
            assert(isinstance(datapoint, bytes) or isinstance(datapoint, memoryview))
            decompressed_datapoint = lz4.block.decompress(datapoint)
            yield decompressed_datapoint


class SnappyCompressData(tensorpack.dataflow.ProxyDataFlow):
    """
    Compresses a datapoint using snappy.
    """

    def __init__(self, ds):
        """
        Args:
            ds (DataFlow): Input dataflow.
        """
        super(SnappyCompressData, self).__init__(ds)

    def get_data(self):
        """
        Yields: Compressed datapoint.
        """
        import snappy
        for datapoint in self.ds.get_data():
            compressed_datapoint = snappy.compress(datapoint, self._compression_mode)
            yield compressed_datapoint


class SnappyDecompressData(tensorpack.dataflow.ProxyDataFlow):
    """
    Decompressed a datapoint using snappy.
    """

    def __init__(self, ds):
        """
        Args:
            ds (DataFlow): Input dataflow.
        """
        super(SnappyDecompressData, self).__init__(ds)

    def get_data(self):
        """
        Yields: Decompressed datapoint.
        """
        import snappy
        for datapoint in self.ds.get_data():
            assert(isinstance(datapoint, bytes))
            decompressed_datapoint = snappy.decompress(datapoint)
            yield decompressed_datapoint


class ZlibCompressData(tensorpack.dataflow.ProxyDataFlow):
    """
    Compresses a datapoint using zlib.
    """

    def __init__(self, ds, compression_level=5):
        """
        Args:
            ds (DataFlow): Input dataflow.
            compression_level (int): Zlib compression level to use
        """
        super(ZlibCompressData, self).__init__(ds)
        self._compression_level = compression_level

    def get_data(self):
        """
        Yields: Compressed datapoint.
        """
        for datapoint in self.ds.get_data():
            compressed_datapoint = zlib.compress(datapoint, self._compression_level)
            yield compressed_datapoint


class ZlibDecompressData(tensorpack.dataflow.ProxyDataFlow):
    """
    Decompressed a datapoint using zlib.
    """

    def __init__(self, ds):
        """
        Args:
            ds (DataFlow): Input dataflow.
        """
        super(ZlibDecompressData, self).__init__(ds)

    def get_data(self):
        """
        Yields: Decompressed datapoint.
        """
        for datapoint in self.ds.get_data():
            assert(isinstance(datapoint, bytes))
            decompressed_datapoint = zlib.decompress(datapoint)
            yield decompressed_datapoint


class Bz2CompressData(tensorpack.dataflow.ProxyDataFlow):
    """
    Compresses a datapoint using bz2.
    """

    def __init__(self, ds, compression_level=5):
        """
        Args:
            ds (DataFlow): Input dataflow.
            compression_level (int): Bz2 compression level to use
        """
        super(Bz2CompressData, self).__init__(ds)
        self._compression_level = compression_level

    def get_data(self):
        """
        Yields: Compressed datapoint.
        """
        for datapoint in self.ds.get_data():
            compressed_datapoint = bz2.compress(datapoint, self._compression_level)
            yield compressed_datapoint


class Bz2DecompressData(tensorpack.dataflow.ProxyDataFlow):
    """
    Decompressed a datapoint using bz2.
    """

    def __init__(self, ds):
        """
        Args:
            ds (DataFlow): Input dataflow.
            compression_level (int): Bz2 compression level to use
        """
        super(Bz2DecompressData, self).__init__(ds)

    def get_data(self):
        """
        Yields: Decompressed datapoint.
        """
        for datapoint in self.ds.get_data():
            assert(isinstance(datapoint, bytes))
            decompressed_datapoint = bz2.decompress(datapoint)
            yield decompressed_datapoint


class BatchDictData(tensorpack.dataflow.ProxyDataFlow):
    """
    Stack datapoints into batches.
    It produces datapoints of the same number of dict entries as ``ds``, but
    each entry has one new extra dimension of size ``batch_size``.
    The entries of the input have to be numpy arrays.
    """

    def __init__(self, ds, batch_size, remainder=False, use_list=False):
        """
        Args:
            ds (DataFlow): When ``use_list=False``, the components of ``ds``
                must be either scalars or :class:`np.ndarray`, and have to be consistent in shapes.
            batch_size(int): batch size
            remainder (bool): When the remaining datapoints in ``ds`` is not
                enough to form a batch, whether or not to also produce the remaining
                data as a smaller batch.
                If set to False, all produced datapoints are guranteed to have the same batch size.
            use_list (bool): if True, each component will contain a list
                of datapoints instead of an numpy array of an extra dimension.
        """
        super(BatchDictData, self).__init__(ds)
        if not remainder:
            try:
                assert batch_size <= ds.size()
            except NotImplementedError:
                pass
        self.batch_size = int(batch_size)
        self.remainder = remainder
        self.use_list = use_list


    def size(self):
        ds_size = self.ds.size()
        div = ds_size // self.batch_size
        rem = ds_size % self.batch_size
        if rem == 0:
            return div
        return div + int(self.remainder)

    def get_data(self):
        """
        Yields:
            Batched data by stacking each entry on an extra 0th dimension.
        """
        holder = []
        for data, in self.ds.get_data():
            holder.append(data)
            if len(holder) == self.batch_size:
                yield [self._aggregate_batch(holder, self.use_list)]
                del holder[:]
        if self.remainder and len(holder) > 0:
            yield [self._aggregate_batch(holder, self.use_list)]


    @staticmethod
    def _aggregate_batch(data_holder, use_list=False):
        size = len(data_holder[0])
        result = {}
        for key in data_holder[0]:
            if use_list:
                result[key] = [x[key] for x in data_holder]
            else:
                dt = data_holder[0][key]
                if type(dt) in [int, bool]:
                    tp = 'int32'
                elif type(dt) == float:
                    tp = 'float32'
                else:
                    try:
                        tp = dt.dtype
                    except:
                        raise TypeError("Unsupported type to batch: {}".format(type(dt)))
                try:
                    result[key] = np.asarray([x[key] for x in data_holder], dtype=tp)
                except KeyboardInterrupt:
                    raise
                except Exception as e:  # noqa
                    logger.exception("Cannot batch data. Perhaps they are of inconsistent shape?")
                    if isinstance(dt, np.ndarray):
                        s = pprint.pformat([x[key].shape for x in data_holder])
                        logger.error("Shape of all arrays to be batched: " + s)
                    try:
                        # open an ipython shell if possible
                        import IPython as IP; IP.embed()    # noqa
                    except:
                        pass
        return result


class UnbatchData(tensorpack.dataflow.ProxyDataFlow):
    """
    Unstack batches into single datapoints.
    It produces datapoints of the same number of components as ``ds``, but
    each component has one less leading dimension.
    The batch has to be a list of numpy arrays with a leading dimension of equal size.
    """

    def __init__(self, ds, batch_size=None, total_size=None):
        """
        Args:
            ds (DataFlow): When ``use_list=False``, the components of ``ds``
                must be :class:`np.ndarray`, and have a leading dimension of equal size.
            batch_size(int or None): size of each batch. If ``None`` then ``size()`` is not supported.
            total_size(int or None): total number of datapoints. Overrides ``batch_size``.
        """
        super(UnbatchData, self).__init__(ds)
        self._batch_size = batch_size
        self._total_size = total_size

    def size(self):
        if self._total_size is not None:
            return self._total_size
        elif self._batch_size is not None:
            return self.ds.size() * self._batch_size
        else:
            raise ValueError("size() is unavailable for unbatching dataflow")

    def get_data(self):
        """
        Yields:
            Single datapoints by removing the leading dimension of each component.
        """
        for data_batch in self.ds.get_data():
            for i in range(len(data_batch[0])):
                datapoint = [batch[i, ...] for batch in data_batch]
                yield datapoint


class UnbatchDictData(tensorpack.dataflow.ProxyDataFlow):
    """
    Unstack batches into single datapoints.
    It produces datapoints of the same number of dict entries as ``ds``, but
    each entry has one less leading dimension.
    The batch has to be a dict of numpy arrays with a leading dimension of equal size.
    """

    def __init__(self, ds, batch_size=None, total_size=None):
        """
        Args:
            ds (DataFlow): When ``use_list=False``, the dict entries of ``ds``
                must be :class:`np.ndarray`, and have a leading dimension of equal size.
            batch_size(int or None): size of each batch. If ``None`` then ``size()`` is not supported.
        """
        super(UnbatchDictData, self).__init__(ds)
        self._batch_size = batch_size
        self._total_size = total_size

    def size(self):
        if self._total_size is not None:
            return self._total_size
        elif self._batch_size is not None:
            return self.ds.size() * self._batch_size
        else:
            raise ValueError("size() is unavailable for unbatching dataflow")

    def get_data(self):
        """
        Yields:
            Single datapoints by removing the leading dimension of each dict entry.
        """
        first_key = None
        for data_batch, in self.ds.get_data():
            if first_key is None:
                first_key = list(data_batch.keys())[0]
            for i in range(len(data_batch[first_key])):
                datapoint = {key: batch[i, ...] for key, batch in data_batch.items()}
                yield [datapoint]


class AutoLMDBData(tensorpack.dataflow.ProxyDataFlow):

    def __init__(self, lmdb_path, shuffle=False, auto_unbatch=True, keys=None, random_start_position=False,
                 use_prefetching=False, finish_prefetching=True, prefetch_queue_size=128, prefetch_count=4,
                 max_num_samples=None):
        self._lmdb_df = LMDBDataWithMetaData(lmdb_path, shuffle, keys, random_start_position,
                                             deserialize_datapoints=False, max_num_samples=max_num_samples)
        self._batch_size = self._lmdb_df.get_metadata("batch_size")
        compression = self._lmdb_df.get_metadata("compression")
        total_size = self._lmdb_df.get_metadata("total_size")
        logger.info("total_size: {}".format(total_size))
        df = self._lmdb_df
        if use_prefetching:
            self._prefetch_df = PrefetchDataZMQ(df, 1, hwm=prefetch_queue_size)
            df = self._prefetch_df
        else:
            self._prefetch_df = None
        # Extract only value and drop key. Directly yield the value and not a list (necessary for ZlibDecompress).
        df = tensorpack.dataflow.MapData(df, lambda datapoint: datapoint[1])
        df = AutoDecompressData(df, compression)
        df = AutoDeserializeData(df, self._lmdb_df.serializer)
        if use_prefetching and finish_prefetching:
            df = PrefetchDataZMQ(df, prefetch_count, hwm=prefetch_queue_size)
        if auto_unbatch:
            df = AutoUnbatchDictData(df, self._batch_size, total_size=total_size)
        super(AutoLMDBData, self).__init__(df)

    def start(self):
        if self._prefetch_df is not None:
            # Make sure we start the LMDB db prefetcher before the other data fetchers. This way all forks
            # start from the main process (this current one) and we can clean them up properly
            self._prefetch_df.start()
            self._lmdb_df.close_lmdb()

    def reset_state(self):
        self.start()
        self.ds.reset_state()

    def total_sample_size(self):
        return self._lmdb_df.size() * self._batch_size

    def batch_size(self):
        return self._batch_size

    def get_keys(self):
        return self._lmdb_df.get_keys()

    def get_item(self, key, deserialize=True):
        return self._lmdb_df.get_item(key, deserialize=deserialize)

    def get_metadata(self, key, deserialize=True):
        return self._lmdb_df.get_metadata(key, deserialize=deserialize)

    def has_metadata(self, key):
        return self._lmdb_df.has_metadata(key)


class DataFlowToTensorflowBridge(thread_utils.CoordinatedThread):

    def __init__(self, input_df, tf_queue, sess, coord, placeholders=None):
        self._input_df = input_df
        self._tf_queue = tf_queue
        self._sess = sess
        self._coord = coord
        self._placeholders = placeholders
        if self._placeholders is not None:
            self._enqueue_op = self._tf_queue.enqueue(self._placeholders)
        elif tf_queue.shapes is not None:
            self._placeholders = [tf.placeholder(dtype=dtype, shape=shape, name="sample_tensor_{}".format(i)) \
                                  for i, (dtype, shape) in enumerate(zip(tf_queue.dtypes, tf_queue.shapes))]
            self._enqueue_op = self._tf_queue.enqueue(self._placeholders)
        else:
            self._enqueue_op = None
        self._close_op = self._tf_queue.close(cancel_pending_enqueues=True)
        super(DataFlowToTensorflowBridge, self).__init__(coord, target=self._run, name="DataFlowToTensorflowBridge")

    def _run(self):
        self._input_df.reset_state()

        # For performance tuning
        # tensors_np = [np.ones(shape=ph.get_shape().as_list(), dtype=np.float32) for ph in self._placeholders]

        try:
            # For performance tuning
            # while True:
            #     self._sess.run(self._enqueue_op, feed_dict=dict(zip(self._placeholders, tensors_np)))

            for data_point in self._input_df.get_data():
                if self.stopped():
                    break
                if self._placeholders is None:
                    self._placeholders = [tf.placeholder(shape=array.shape, dtype=array.dtype) for array in data_point]
                    self._enqueue_op = self._tf_queue.enqueue(self._placeholders)
                self._sess.run(self._enqueue_op, feed_dict=dict(zip(self._placeholders, data_point)))
        except (tf.errors.CancelledError, tf.errors.OutOfRangeError, tensorpack.dataflow.DataFlowTerminated):
            pass
        except Exception as e:
            logger.exception(e)
        finally:
            self._sess.run(self._close_op)

    def start(self):
        self.daemon = True
        super(DataFlowToTensorflowBridge, self).start()

    def stop(self):
        self._thread.stop()

    @property
    def thread(self):
        return self._thread


class BatchDataFlowToTensorflowBridge(thread_utils.CoordinatedThread):

    def __init__(self, batch_input_df, tf_queue, sess, coord, placeholders=None):
        self._batch_input_df = batch_input_df
        self._tf_queue = tf_queue
        self._sess = sess
        self._coord = coord
        self._placeholders = placeholders
        if self._placeholders is not None:
            self._enqueue_op = self._tf_queue.enqueue_many(self._placeholders)
        elif tf_queue.shapes is not None:
            self._placeholders = [tf.placeholder(dtype=dtype, shape=[None] + shape.as_list(),
                                                 name="sample_tensor_{}".format(i))
                                  for i, (dtype, shape) in enumerate(zip(tf_queue.dtypes, tf_queue.shapes))]
            self._enqueue_op = self._tf_queue.enqueue_many(self._placeholders)
        else:
            self._enqueue_op = None
        self._close_op = self._tf_queue.close(cancel_pending_enqueues=True)
        super(BatchDataFlowToTensorflowBridge, self).__init__(coord, target=self._run, name="BatchDataFlowToTensorflowBridge")

    def _run(self):
        self._batch_input_df.reset_state()

        # For performance tuning
        # tensors_np = [np.ones(shape=ph.get_shape().as_list(), dtype=np.float32) for ph in self._placeholders]

        try:

            # For performance tuning
            # while True:
            #     self._sess.run(self._enqueue_op, feed_dict=dict(zip(self._placeholders, tensors_np)))

            for batch in self._batch_input_df.get_data():
                if self.stopped():
                    break
                if self._placeholders is None:
                    logger.info("Creating placeholders on the fly")
                    self._placeholders = [tf.placeholder(shape=array.shape, dtype=array.dtype) for array in batch]
                    self._enqueue_op = self._tf_queue.enqueue_many(self._placeholders)
                self._sess.run(self._enqueue_op, feed_dict=dict(zip(self._placeholders, batch)))
        except (tf.errors.CancelledError, tf.errors.OutOfRangeError, tensorpack.dataflow.DataFlowTerminated):
            pass
        except Exception as e:
            logger.exception(e)
        finally:
            # Try to close the queue but ignore any RuntimeErrors (i.e. closed session)
            try:
                self._sess.run(self._close_op)
            except RuntimeError as exc:
                pass

    def start(self):
        self.daemon = True
        super(BatchDataFlowToTensorflowBridge, self).start()

    @property
    def thread(self):
        return self


import itertools
from tensorpack.utils.concurrency import (ensure_proc_terminate, mask_sigint, start_proc_mask_signal)


def _bind_guard(sock, name):
    try:
        sock.bind(name)
    except zmq.ZMQError:
        logger.error(
            "ZMQError in socket.bind(). Perhaps you're \
            using pipes on a non-local file system. See documentation of PrefetchDataZMQ for more information.")
        raise


def _get_pipe_name(name):
    pipedir = os.environ.get('TENSORPACK_PIPEDIR', '.')
    assert os.path.isdir(pipedir), pipedir
    pipename = "ipc://{}/{}-pipe-".format(pipedir.rstrip('/'), name) + str(uuid.uuid1())
    return pipename


@contextmanager
def _zmq_catch_error(name):
    try:
        yield
    except zmq.ContextTerminated:
        logger.info("[{}] Context terminated.".format(name))
        raise DataFlowTerminated()
    except zmq.ZMQError as e:
        if e.errno == errno.ENOTSOCK:       # socket closed
            logger.info("[{}] Socket closed.".format(name))
            raise DataFlowTerminated()
        else:
            raise
    except:
        raise


class _MultiProcessZMQDataFlow(tensorpack.dataflow.DataFlow):

    def __init__(self, ds):
        assert os.name != 'nt', "ZMQ IPC doesn't support windows!"
        self._start_done = False
        self._reset_done = False
        self._procs = []

        self.ds = ds
        try:
            self._size = ds.size()
        except NotImplementedError:
            self._size = -1

    def size(self):
        return self.ds.size()

    def start(self):
        if self._start_done:
            return
        self._start_done = True

        self._start_once()

        # __del__ not guranteed to get called at exit
        import atexit
        atexit.register(lambda x: x.__del__(), self)

    def _start_once(self):
        pass

    def reset_state(self):
        """
        All forked dataflows are reset **once and only once** in spawned processes.
        Nothing more can be done when calling this method.
        """
        self.start()
        if self._reset_done:
            return
        self._reset_done = True

        self._reset_once()  # build processes

    def _reset_once(self):
        pass

    def _start_processes(self):
        logger.info("Starting {} processes (pid={})".format(len(self._procs), os.getpid()))
        start_proc_mask_signal(self._procs)

    def __del__(self):
        if self._start_done:
            for x in self._procs:
                x.terminate()
        if self._reset_done:
            if not self.context.closed:
                self.context.destroy(0)
        try:
            print("{} successfully cleaned-up.".format(type(self).__name__))
        except:
            pass


class PrefetchDataZMQ(_MultiProcessZMQDataFlow):
    """
    Prefetch data from a DataFlow using multiple processes, with ZeroMQ for
    communication.
    It will fork the calling process of :meth:`reset_state()`,
    and collect datapoints from `ds` in each process by ZeroMQ IPC pipe.

    Note:
        1. An iterator cannot run faster automatically -- what's happenning is
           that the underlying dataflow will be forked ``nr_proc`` times.
           As a result, we have the following guarantee on the dataflow correctness:

           a. When ``nr_proc=1``, the dataflow produces the same data as ``ds`` in the same order.
           b. When ``nr_proc>1``, the dataflow produces the same distribution
              of data as ``ds`` if each sample from ``ds`` is i.i.d. (e.g. fully shuffled).
              You probably only want to use it for training.

        2. Once :meth:`reset_state` is called, this dataflow becomes not fork-safe.
           i.e., if you fork an already reset instance of this dataflow,
           it won't be usable in the forked process.
        3. When nesting like this: ``PrefetchDataZMQ(PrefetchDataZMQ(df, nr_proc=a), nr_proc=b)``.
           A total of ``a * b`` instances of ``df`` worker processes will be created.
           Also in this case, some zmq pipes cannot be cleaned at exit.
        4. By default, a UNIX named pipe will be created in the current directory.
           However, certain non-local filesystem such as NFS/GlusterFS/AFS doesn't always support pipes.
           You can change the directory by ``export TENSORPACK_PIPEDIR=/other/dir``.
           In particular, you can use somewhere under '/tmp' which is usually local.

           Note that some non-local FS may appear to support pipes and code
           may appear to run but crash with bizarre error.
           Also note that ZMQ limits the maximum length of pipe path.
           If you hit the limit, you can set the directory to a softlink
           which points to a local directory.
        5. Calling `reset_state()` more than once is a no-op, i.e. the worker processes won't get called.
    """

    class _Worker(multiprocessing.Process):
        def __init__(self, ds, pipename, hwm, bind, serializer):
            super(PrefetchDataZMQ._Worker, self).__init__()
            self.ds = ds
            self.pipename = pipename
            self.hwm = hwm
            self.bind = bind
            self.serializer = serializer

        def run(self):
            logger.info("Starting worker for pipe={}, pid={}".format(self.pipename, os.getpid()))
            self.ds.reset_state()
            context = zmq.Context()
            socket = context.socket(zmq.PUSH)
            socket.set_hwm(self.hwm)
            if self.bind:
                _bind_guard(socket, self.pipename)
            else:
                socket.connect(self.pipename)
            try:
                while True:
                    for dp in self.ds.get_data():
                        # logger.debug("Pushing ZMQ message (pid={})".format(os.getpid()))
                        socket.send(self.serializer.dumps(dp), copy=False)
            # sigint could still propagate here, e.g. when nested
            except KeyboardInterrupt:
                pass

    def __init__(self, ds, nr_proc=1, hwm=50, serializer=None):
        """
        Args:
            ds (DataFlow): input DataFlow.
            nr_proc (int): number of processes to use.
            hwm (int): the zmq "high-water mark" (queue size) for both sender and receiver.
        """
        super(PrefetchDataZMQ, self).__init__(ds)

        if serializer is None:
            serializer = serialization.MsgPackSerializer()
        self._serializer = serializer

        self.nr_proc = nr_proc
        self._hwm = hwm

        self._pipename = _get_pipe_name('dataflow')
        self._guard = DataFlowReentrantGuard()

    def _recv(self):
        return self._serializer.loads(self.socket.recv(copy=False).bytes)

    def get_data(self):
        with self._guard, _zmq_catch_error('PrefetchDataZMQ'):
            for k in itertools.count():
                if self._size > 0 and k >= self._size:
                    break
                # logger.debug("Waiting for ZMQ message (pid={})".format(os.getpid()))
                yield self._recv()

    def _start_once(self):
        logger.info("Resetting ZMQ prefetcher (pid={}, pipe={})".format(os.getpid(), self._pipename))
        worker_bind = self.nr_proc == 1
        self._procs = [PrefetchDataZMQ._Worker(self.ds, self._pipename, self._hwm, worker_bind, self._serializer)
                       for _ in range(self.nr_proc)]
        self._start_processes()

    def _reset_once(self):
        logger.info("Resetting ZMQ prefetcher connection (pid={}, pipe={})".format(os.getpid(), self._pipename))
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.set_hwm(self._hwm)
        if self.nr_proc == 1:
            self.socket.connect(self._pipename)
        else:
            _bind_guard(self.socket, self._pipename)

    @property
    def processes(self):
        return self._procs

    # @staticmethod
    # def cleanup(self):
    #     import os
    #     import signal
    #     import psutil
    #     current_process = psutil.Process()
    #     children = current_process.children(recursive=True)
    #     for child in children:
    #         logger.info('Child pid is {}'.format(child.pid))
    #         try:
    #             os.kill(child.pid, signal.SIGTERM)
    #         except:
    #             pass
    #     # for proc in self._all_procs:
    #     #     try:
    #     #         proc.terminate()
    #     #     except:
    #     #         pass
