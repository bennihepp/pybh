import threading


class StoppableThread(threading.Thread):

    def __init__(self, *args, **kwargs):
        self._stop_event = kwargs.pop("stop_event", None)
        if self._stop_event is None:
            self._stop_event = threading.Event()
        super(StoppableThread, self).__init__(*args, **kwargs)

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.isSet()


class CoordinatedThread(StoppableThread):

    def __init__(self, coord, *args, **kwargs):
        # Allow a tensorflow Coordinator object
        if not callable(coord):
            coord = coord.should_stop
        self._coord = coord
        super(CoordinatedThread, self).__init__(*args, **kwargs)

    def stopped(self):
        return self._coord() or super(CoordinatedThread, self).stopped()


class LoopThread(StoppableThread):

    def __init__(self, iteration_func):
        self._iteration_func = iteration_func
        super(LoopThread, self).__init__(target=self._run)

    def _run(self):
        while not self.stopped():
            self._iteration_func()


class CoordinatedLoopThread(CoordinatedThread):

    def __init__(self, iteration_func, coord, *args, **kwargs):
        self._iteration_func = iteration_func
        super(CoordinatedLoopThread, self).__init__(coord, *args, **kwargs)

    def _run(self):
        while not self.stopped():
            self._iteration_func()


global __profiling_stats__
global __profiling_stats_lock__
global __profiling_stats_filename__
__profiling_stats__ = None
__profiling_stats_lock__ = None
__profiling_stats_filename__ = None


def enable_thread_profiling(profiling_stats_filename):
    """Thanks to http://rjp.io/2013/05/16/thread-profiling-in-python/ for this function.

    Monkey-patch Thread.run to enable global profiling.
    Each thread creates a local profiler; statistics are pooled
    to the global stats object on run completion."""
    import cProfile
    import pstats

    global __profiling_stats_lock__
    global __profiling_stats_filename__
    __profiling_stats_lock__ = threading.Lock()
    __profiling_stats_filename__ = profiling_stats_filename

    threading.Thread.__original_run__ = threading.Thread.run

    def run_with_profiling(self):
        global __profiling_stats__
        global __profiling_stats_lock__
        global __profiling_stats_filename__

        self._prof = cProfile.Profile()
        self._prof.enable()
        threading.Thread.__original_run__(self)
        self._prof.disable()

        with __profiling_stats_lock__:
            if __profiling_stats__ is None:
                __profiling_stats__ = pstats.Stats(self._prof)
            else:
                __profiling_stats__.add(self._prof)
            if __profiling_stats_filename__ is not None:
                __profiling_stats__.dump_stats(__profiling_stats_filename__)

    threading.Thread.run = run_with_profiling


def disable_thread_profiling():
    global __profiling_stats__
    global __profiling_stats_lock__
    if __profiling_stats__ is None:
        raise RuntimeError("Thread profiling has not been enabled or no threads have finished running.")
    threading.Thread.run = threading.Thread.__original_run__
    del threading.Thread.__original_run__
    __profiling_stats__ = None
    __profiling_stats_lock__ = None


def get_thread_profiling_stats():
    global __profiling_stats__
    if __profiling_stats__ is None:
        raise RuntimeError("Thread profiling has not been enabled or no threads have finished running.")
    return __profiling_stats__

