import fire

from . import config, clients


def submit(sid, n_estimators=100):
    getattr(clients, 'submit_' + str(sid))(n_estimators)


def main():
    config.setup_logger()

    fire.Fire({
        'submit': submit
    })
