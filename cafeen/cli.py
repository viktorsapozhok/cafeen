import fire

from . import config, clients


def encode_files():
    clients.encode_files()


def submit(sid):
    getattr(clients, 'submit_' + str(sid))()


def main():
    config.setup_logger()

    fire.Fire({
        'encode': encode_files,
        'submit': submit
    })
