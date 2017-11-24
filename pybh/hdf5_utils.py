from __future__ import print_function

import os
import time
import numpy as np
import threading
import multiprocessing
try:
    from queue import Queue, Full, Empty
except ImportError:
    from Queue import Queue, Full, Empty
try:
    import cPickle as pickle
except ImportError:
    import pickle
import h5py


def write_attribute_dict_to_hdf5_group(group, attr_dict):
    for key, value in attr_dict.items():
        if isinstance(value, dict):
            subgroup = group.require_group(key)
            write_attribute_dict_to_hdf5_group(subgroup, value)
        else:
            try:
                group.attrs[key] = value
            except TypeError:
                print("Failed to set attribute for key {}".format(key))
                raise


def write_numpy_dict_to_hdf5_group(group, numpy_dict, **dataset_kwargs):
    for key, value in numpy_dict.items():
        if key == "__attrs__":
          assert(isinstance(value, dict))
          write_attribute_dict_to_hdf5_group(group, value)
        elif isinstance(value, dict):
            subgroup = group.require_group(key)
            write_numpy_dict_to_hdf5_group(subgroup, value, **dataset_kwargs)
        else:
            array = np.asarray(value)
            # Just let h5py deal with the datatype
            # if array.dtype == np.float:
            #     hdf5_dtype = 'd'
            # elif array.dtype == np.float32:
            #     hdf5_dtype = 'f'
            # elif array.dtype == np.float64:
            #     hdf5_dtype = 'd'
            # elif array.dtype == np.int:
            #     hdf5_dtype = 'i8'
            # elif array.dtype == np.int32:
            #     hdf5_dtype = 'i4'
            # elif array.dtype == np.int64:
            #     hdf5_dtype = 'i8'
            # else:
            #     raise NotImplementedError("Unsupported datatype for write_hdf5_file() helper: {}".format(array.dtype))
            # dataset = f.create_dataset(key, shape=array.shape, dtype=array.dtype, data=array)
            # dataset[...] = array
            try:
                group.create_dataset(key, data=array, **dataset_kwargs)
            except TypeError:
                print("Failed to create dataset for key {}".format(key))
                raise


def write_numpy_dict_to_hdf5_file(filename, numpy_dict, attr_dict=None, **dataset_kwargs):
    f = h5py.File(filename, "w")
    try:
        write_numpy_dict_to_hdf5_group(f, numpy_dict, **dataset_kwargs)
        if attr_dict is not None:
            write_attribute_dict_to_hdf5_group(f, attr_dict)
    finally:
        f.close()


def read_hdf5_group_to_attr_dict(group, attr_dict):
    for key, value in group.attrs.items():
        attr_dict[key] = value
    for key in group:
        if isinstance(group[key], h5py.Group):
            sub_attr_dict = {}
            read_hdf5_group_to_attr_dict(group[key], sub_attr_dict)
            if len(sub_attr_dict) > 0:
                attr_dict[key] = sub_attr_dict


def read_hdf5_group_to_numpy_dict(group, numpy_dict, field_dict=None):
    for key in group:
        if field_dict is not None and key not in field_dict:
            continue
        if isinstance(group[key], h5py.Group):
            numpy_dict[key] = {}
            sub_field_dict = None
            if field_dict is not None:
                sub_field_dict = field_dict[key]
            read_hdf5_group_to_numpy_dict(group[key], numpy_dict[key], sub_field_dict)
        else:
            numpy_dict[key] = np.array(group[key])
            if field_dict is not None:
                field_dict[key] = True


def read_hdf5_file_to_numpy_dict(filename, field_dict=None, read_attributes=False):
    f = h5py.File(filename, "r")
    try:
        numpy_dict = {}
        read_hdf5_group_to_numpy_dict(f, numpy_dict, field_dict)
        if read_attributes:
            attr_dict = {}
            read_hdf5_group_to_attr_dict(f, attr_dict)
            return numpy_dict, attr_dict
        return numpy_dict
    finally:
        f.close()


class QueueEnd(object):

    def __eq__(self, other):
        """Make sure that QueueEnd is the same across different processes"""
        return type(self) == type(other)


QUEUE_END = QueueEnd()


class HDF5ReaderProcess(object):

    def __init__(self, filename_queue, data_pipe_out, signal_queue,
                 read_data_fn, timeout, verbose=False, _id=None,
                 max_send_size=1024 * 1024 * 1024,):
        self._filename_queue = filename_queue
        self._data_pipe_out = data_pipe_out
        self._signal_queue = signal_queue
        self._read_data_fn = read_data_fn
        self._timeout = timeout
        self._process = None
        self._verbose = verbose
        self._id = _id
        self._keep_running = False
        self._max_send_size = max_send_size

    def _should_keep_running(self):
        if self._keep_running:
            try:
                sig = self._signal_queue.get_nowait()
                if sig == QUEUE_END:
                    self._keep_running = False
            except Empty:
                pass
        return self._keep_running

    def _parent_alive(self):
        if self._keep_running:
            ppid = os.getppid()
            if ppid == 1:
                print("Parent process has terminated. Stopping.")
                self._keep_running = False
        return self._keep_running

    def _run(self):
        self._filename_queue.cancel_join_thread()
        # self._data_queue.cancel_join_thread()
        self._signal_queue.cancel_join_thread()
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        # Reopen stdout and stderr without buffering
        # import sys, os
        # sys.stdout = os.fdopen(sys.stdout.fileno(), "a", 0)
        # sys.stderr = os.fdopen(sys.stderr.fileno(), "a", 0)
        if self._verbose:
            print("Entering HDF5 queue reader process {}".format(self._id if self._id is not None else "<unnamed>"))
        data_count = 0
        while self._should_keep_running() and self._parent_alive():
            try:
                # filename = self._filename_queue.get(block=True)
                filename = self._filename_queue.get(block=True, timeout=self._timeout)
            except Empty:
                continue
            # if filename is None:
            #     break
            if filename == QUEUE_END:
                if self._verbose:
                    print("HDF5 filename queue finished. Pushing QUEUE_END signal into data queue.")
                end_dump = pickle.dumps(QUEUE_END, pickle.HIGHEST_PROTOCOL)
                self._data_pipe_out.send_bytes(end_dump)
                break
            data = self._read_data_fn(filename)
            for data_slice_dump in self._data_slice_dump_generator(data):
                enqueued = False
                while not enqueued:
                    try:
                        # self._data_queue.put(records_dump, block=True, timeout=self._timeout)
                        if self._verbose:
                            print("HDF5ReaderProcess: Sending data block")
                        self._data_pipe_out.send_bytes(data_slice_dump)
                        enqueued = True
                    except Full:
                        pass
                    if not self._parent_alive():
                        break
                if not self._parent_alive():
                    break
            data_count += 1
            if self._verbose >= 2:
                print("Enqueued {} data chunks.".format(data_count))
        if self._verbose:
            print("Stop request... Exiting HDF5 queue reader process {}".format(
                self._id if self._id is not None else "<unnamed>"))

    def _data_slice_dump_generator(self, data, idx_from=0):
        slice_size = self._max_send_size
        first_key = next(iter(data.keys()))
        idx_offset = len(data[first_key])
        data_slice_dump = pickle.dumps(data, pickle.HIGHEST_PROTOCOL)
        data_slice_size = len(data_slice_dump)
        while data_slice_size > self._max_send_size:
            idx_offset //= 2
            if idx_offset == 0:
                raise ValueError("Unable to slice data object to a size <= {} bytes".format(self._max_send_size))
            idx_to = idx_from + idx_offset
            data_slice = {key: value[idx_from:idx_to] for key, value in data.items()}
            data_slice_dump = pickle.dumps(data_slice, pickle.HIGHEST_PROTOCOL)
            data_slice_size = len(data_slice_dump)
        yield data_slice_dump
        idx_from += idx_offset
        if idx_from < len(data[first_key]):
            for data_slice_dump in self._data_slice_dump_generator(data, idx_from=idx_from):
                yield data_slice_dump

    def start(self):
        assert (self._process is None)
        self._keep_running = True
        self._process = multiprocessing.Process(target=self._run, name="HDF5ReaderProcess")
        self._process.daemon = True
        self._process.start()

    def is_alive(self):
        if self._process is None:
            return False
        else:
            return self._process.is_alive()

    def join(self):
        if self._process is not None:
            self._process.join()

    @property
    def process(self):
        return self._process


class HDF5ReaderProcessCoordinator(object):

    def __init__(self, filenames, coord, read_data_fn, shuffle=False,
                 repeats=-1, timeout=60, num_processes=1, data_queue_capacity=10, verbose=False):
        num_processes = min(num_processes, len(filenames))
        self._filenames = list(filenames)
        self._coord = coord
        self._read_data_fn = read_data_fn
        self._shuffle = shuffle
        self._repeats = repeats
        self._timeout = timeout
        self._num_processes = num_processes
        self._data_queue_capacity = data_queue_capacity
        self._verbose = verbose

        self._filename_push_thread = None
        self._data_pull_thread = None

        self._initialize_pipe_and_readers()
        self._data_queue = Queue(data_queue_capacity)

    def _initialize_pipe_and_readers(self):
        self._data_pipe_ins = []
        self._data_pipe_outs = []
        for i in range(self._num_processes):
            data_pipe_out, data_pipe_in = multiprocessing.Pipe(duplex=False)
            self._data_pipe_outs.append(data_pipe_out)
            self._data_pipe_ins.append(data_pipe_in)
        self._filename_queue = multiprocessing.Queue(maxsize=2 * len(self._filenames))
        self._signal_queues = [multiprocessing.Queue() for _ in range(self._num_processes)]
        reader_timeout = self._timeout / 10.
        self._reader_processes = [
            HDF5ReaderProcess(self._filename_queue, self._data_pipe_ins[i], self._signal_queues[i],
                              self._read_data_fn, reader_timeout,
                              self._verbose, _id=i)
            for i in range(self._num_processes)]

    def _start_readers(self):
        for reader in self._reader_processes:
            assert(not reader.is_alive())
            reader.start()

    def _stop_readers(self):
        for signal_queue in self._signal_queues:
            signal_queue.put(QUEUE_END)
            signal_queue.close()

        for data_pipe_in in self._data_pipe_ins:
            data_pipe_in.close()
        for data_pipe_out in self._data_pipe_outs:
            data_pipe_out.close()

        self._filename_queue.close()
        # self._data_queue.cancel_join_thread()
        self._filename_queue.cancel_join_thread()
        for signal_queue in self._signal_queues:
            signal_queue.cancel_join_thread()

        if self._verbose:
            print("Terminating HDF5ReaderProcesses")
        for reader in self._reader_processes:
            reader.process.terminate()
            # reader.process.join()
        # import os, signal
        # for reader in self._reader_processes:
        #     os.kill(reader.process.pid, signal.SIGKILL)
        import sys
        sys.stdout.flush()
        if self._verbose:
            print("Resetting queue and HDF5ReaderProcesses")
        self._initialize_pipe_and_readers()

    def _run_push_filenames(self):
        if self._verbose:
            print("Entering HDF5ReaderProcessCoordinator push filenames thread")
        num_repeats = 0
        while not self._coord.should_stop():
            if self._shuffle:
                np.random.shuffle(self._filenames)
            for filename in self._filenames:
                enqueued = False
                while not enqueued:
                    try:
                        self._filename_queue.put(filename, block=True, timeout=self._timeout)
                        enqueued = True
                    except Full:
                        pass
                    if self._coord.should_stop():
                        break
                if self._coord.should_stop():
                    break
            num_repeats += 1
            if self._repeats > 0 and num_repeats >= self._repeats:
                break
        if self._verbose:
            print("Stop request... Exiting HDF5ReaderProcessCoordinator push filenames thread")
        for _ in self._reader_processes:
            self._filename_queue.put(QUEUE_END)

    def _run_pull_data(self):
        if self._verbose:
            print("Entering HDF5ReaderProcessCoordinator pull data thread")
        breakout = False
        queue_end_count = 0
        while not self._coord.should_stop():
            received_something = False
            for data_pipe_out in self._data_pipe_outs:
                if data_pipe_out.poll():
                    data_dump = data_pipe_out.recv_bytes()
                    data = pickle.loads(data_dump)
                    if data == QUEUE_END:
                        if self._verbose:
                            print("HDF5ReaderProcessCoordinator: Received QUEUE_END signal")
                        queue_end_count += 1
                        if queue_end_count >= len(self._reader_processes):
                            breakout = True
                            break
                    self._data_queue.put(data)
                    received_something = True
            if breakout:
                break
            if not received_something:
                time.sleep(0.01)
        self._data_queue.put(QUEUE_END)
        if self._verbose:
            print("Stop request... Exiting HDF5ReaderProcessCoordinator pull data thread")

    def get_next_data(self):
        # while len(self._local_record_dequeue) == 0:
        #     time.sleep(0.01)
            # from pybh.utils import Timer
            # timer = Timer()
            # records = self.get_next_record_batch()
            # self._local_record_dequeue.extendleft(records)
            # print("Retrieved batch of records from pipe. Took {}s".format(timer.elapsed_seconds()))
        return self._data_queue.get()

    # def get_next_record_batch(self):
    #     while not self._coord.should_stop():
    #         try:
    #             # records_dump = self._data_queue.get(block=True, timeout=self._timeout)
    #             records_dump = self._record_pipe_ins
    #             records = pickle.loads(records_dump)
    #             return records
    #         except Empty:
    #             pass
    #     if self._verbose:
    #         print("Stop request... Exiting HDF5ReaderProcessCoordinator queue thread")
    #     raise StopIteration()

    def stop(self):
        # Terminate reader processes
        self._stop_readers()

    def start(self):
        assert (self._filename_push_thread is None)
        assert (self._data_pull_thread is None)
        self._start_readers()

        self._filename_push_thread = threading.Thread(target=self._run_push_filenames,
                                                      name="HDF5ReaderProcessCoordinator.push_filenames")
        self._filename_push_thread.daemon = True
        self._filename_push_thread.start()
        self._data_pull_thread = threading.Thread(target=self._run_pull_data,
                                                    name="HDF5ReaderProcessCoordinator.pull_data")
        self._data_pull_thread.daemon = True
        self._data_pull_thread.start()

    @property
    def threads(self):
        return [self._filename_push_thread, self._data_pull_thread]

    @property
    def filename_push_thread(self):
        return self._filename_push_thread

    @property
    def data_pull_thread(self):
        return self._data_pull_thread

    @property
    def filename_queue(self):
        return self._filename_queue

    @property
    def data_queue(self):
        return self._data_queue

