import sys
import time
import argparse


def fail(message, exit_code=1):
  sys.stderr.write(message + "\n")
  sys.exit(exit_code)


def argparse_bool(v):
    if type(v) == str:
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    elif type(v) == bool:
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

