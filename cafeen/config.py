import logging
from os import path

from colorlog import ColoredFormatter

root_dir = path.join(path.abspath(path.dirname(__file__)), '..')

path_to_data = path.join(root_dir, 'data')
path_to_train = path.join(path_to_data, 'train.csv')
path_to_test = path.join(path_to_data, 'test.csv')


def setup_logger():
    logger = logging.getLogger('cafeen')

    stream_formatter = ColoredFormatter(
        '%(log_color)s%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        reset=True,
        log_colors={
            'DEBUG': 'green',
            'INFO': 'black',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red'
        }
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)

    logger.setLevel(logging.DEBUG)

    return logger
