import sys
import time
import argparse
from builtins import dict
from functools import reduce
from contextlib import contextmanager
import six


def fail(message, exit_code=1):
    sys.stderr.write(message + "\n")
    sys.exit(exit_code)


def convert_string_to_array(arg, sep=None, value_type=None, value_fn=None, do_strip=True):
    array = [x.strip() for x in arg.split(sep)]
    if do_strip:
        array = [x.strip() for x in array]
    if value_type is not None:
        array = [value_type(x) for x in array]
    if value_fn is not None:
        array = [value_fn(x) for x in array]
    return array


def argparse_bool(v):
    if isinstance(v, bytes):
        v = v.decode("ascii")
    if isinstance(v, six.string_types):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    elif isinstance(v, bool):
        return v
    else:
        raise ValueError("Expected string or boolean value. Got type {}".format(type(v)))


def print_debug(variable_name):
    frame = sys._getframe(1)
    print("{}={}".format(variable_name, eval(variable_name, frame.f_globals, frame.f_locals)))


class Timer(object):

    def __init__(self):
        self._t0 = time.time()

    def restart(self):
        t1 = time.time()
        dt = t1 - self._t0
        self._t0 = time.time()
        return dt

    def elapsed_seconds(self):
        dt = time.time() - self._t0
        return dt

    def print_elapsed_seconds(self, name):
        dt = self.elapsed_seconds()
        print("{} took {}s".format(name, dt))

    def compute_rate(self, num_samples):
        rate = num_samples / float(self.elapsed_seconds())
        return rate

    def print_rate(self, num_samples, name=None, unit="Hz"):
        rate = self.compute_rate(num_samples)
        if name is None:
            print("Running with {} {:s}".format(rate, unit))
        else:
            print("{} running with {} {:s}".format(name, rate, unit))


class RateTimer(Timer):

    def __init__(self, reset_interval=100):
        super(RateTimer, self).__init__()
        self._count = 0
        self._rate = 0
        self._reset_interval = reset_interval

    def update(self, n=1):
        self._count += n
        self._rate = super(RateTimer, self).compute_rate(self._count)
        if self._count >= self._reset_interval:
            self.restart()
            self._count = 0

    def compute_rate(self, num_samples=None):
        if num_samples is None:
            num_samples = self._count
        return super(RateTimer, self).compute_rate(self._count)

    def print_rate(self, name=None, unit="Hz"):
        rate = self._rate
        if name is None:
            print("Running with {} {:s}".format(rate, unit))
        else:
            print("{} running with {} {:s}".format(name, rate, unit))

    def update_and_print_rate(self, name=None, unit="Hz", n=1, print_interval=10):
        self.update(n)
        if self._count % print_interval == 0:
            self.print_rate(name, unit)


class DummyTimeMeter(object):

    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

    def measure(self, key):
        return self

    def start_measurement(self, key):
        pass

    def finish_measurement(self, key):
        pass

    def add_time_measurement(self, key, dt):
        pass

    def get_keys(self):
        return []

    def get_total_time(self, key):
        return 0

    def get_total_times(self, key):
        return []

    def get_time_ratio(self, key):
        return 0

    def get_time_ratios(self, key):
        return []

    def print_times(self):
        pass


class TimeMeter(object):

    def __init__(self, name=None):
        self._name = name
        self._measured_times = dict()
        self._timers = dict()
        self._active_measurement_key = None

    def _get_timer(self, key):
        if key in self._timers:
            return self._timers[key]
        else:
            timer = Timer()
            self._timers[key] = timer
            return timer

    @contextmanager
    def measure(self, key):
        self.start_measurement(key)
        yield
        self.finish_measurement(key)

    def start_measurement(self, key):
        assert self._active_measurement_key is None,\
            "Another measurement ({}) is already running".format(self._active_measurement_key)
        self._get_timer(key).restart()
        self._active_measurement_key = key

    def finish_measurement(self, key):
        assert self._active_measurement_key is not None, "No measurement has been started"
        assert self._active_measurement_key == key,\
            "The started measurement ({}) does not match the key ({})".format(self._active_measurement_key, key)
        dt = self._get_timer(key).elapsed_seconds()
        self.add_time_measurement(key, dt)
        self._active_measurement_key = None

    def add_time_measurement(self, key, dt):
        if key in self._measured_times:
            self._measured_times[key] += dt
        else:
            self._measured_times[key] = dt

    def get_keys(self):
        return self._measured_times.keys()

    def get_total_time(self, key):
        return self._measured_times[key]

    def get_total_times(self):
        return self._measured_times

    def get_time_ratio(self, key):
        return self._measured_times[key] / reduce(lambda x, y: x + y, self._measured_times.values())

    def get_time_ratios(self):
        summed_time = reduce(lambda x, y: x + y, self._measured_times.values())
        return {key: self._measured_times[key] / summed_time for key in self._measured_times}

    def print_times(self):
        if self._name is None:
            print("Timings:")
        else:
            print("Timings for {}:".format(self._name))
        for key in self._measured_times:
            print("  {:s} took {:3f}s ({:2f} %)".format(key, self.get_total_time(key), 100 * self.get_time_ratio(key)))


global __global_time_meters__
__global_time_meters__ = {}


def get_time_meter(name="__main__"):
    global __global_time_meters__
    if name not in __global_time_meters__:
        meter_name = name
        if name == "__main__":
            meter_name = None
        __global_time_meters__[name] = TimeMeter(meter_name)
    return __global_time_meters__[name]


@contextmanager
def time_measurement(msg, log_start=False):
    if log_start:
        print("Starting {} ...".format(msg))
    start_time = time.time()
    yield
    print('Finished {} in {:.3f}s.'.format(msg, time.time() - start_time))


@contextmanager
def logged_time_measurement(logger, msg, log_start=True, format=False):
    if log_start:
        if format:
            logger.info("Starting {} ...".format(msg))
        else:
            logger.info(msg)
    start_time = time.time()
    yield
    logger.info('Finished {} in {:.3f}s.'.format(msg, time.time() - start_time))


class Callbacks(object):

    def __init__(self, return_first_value=False):
        self._callbacks = []
        self._callback_args = []
        self._return_first_value = return_first_value

    def register(self, cb, *args, **kwargs):
        self._callbacks.append(cb)
        self._callback_args.append((args, kwargs))

    def deregister(self, cb):
        idx = self._callbacks.index(cb)
        if idx < 0:
            raise ValueError("Callback not found")
        del self._callbacks[idx]
        del self._callback_args[idx]

    def clear(self, cb):
        self._callbacks = []
        self._callback_args = []

    def __call__(self, *args, **kwargs):
        return_values = []
        for i, cb in enumerate(self._callbacks):
            cb_args, cb_kwargs = self._callback_args[i]
            args = args + cb_args
            kwargs.update(cb_kwargs)
            return_values.append(cb(*args, **kwargs))
        if self._return_first_value:
            if len(return_values) > 0:
                return return_values[0]
            else:
                return None
        else:
            return return_values


@contextmanager
def DummyContextManager():
    pass
    yield
    return
