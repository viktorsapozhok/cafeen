import fire

from . import config, clients


def submit(sid, **kwargs):
    getattr(clients, 'submit_' + str(sid))(**kwargs)


def main():
    config.setup_logger()

    fire.Fire({
        'submit': submit
    })
