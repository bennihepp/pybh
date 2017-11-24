import sys
import logging


def get_logger(name, log_level=logging.INFO):
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(log_level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s > %(message)s"))
    logger.addHandler(handler)
    return logger


class LoggerStream(object):

    def __init__(self, logger, log_level=logging.INFO):
        self._logger = logger
        self._log_level = log_level

    def set_log_level(self, log_level):
        self._log_level = log_level

    def write(self, buf):
        # for line in buf.rstrip().splitlines():
        self._logger.log(self._log_level, buf.rstrip())

    def flush(self):
        pass


def get_logger_stream(name, log_level=logging.INFO):
    logger = get_logger(name, log_level)
    stream = LoggerStream(logger, log_level)
    return stream
