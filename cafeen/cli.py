import fire

from . import config, clients


def encode_files():
    config.setup_logger()

    clients.encode_files()


def main():
    fire.Fire({
        'encode': encode_files
    })
