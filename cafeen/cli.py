import fire

from . import config, clients


def submit(sid, n_estimators=100, eta=0.1, nrows=None):
    getattr(clients, 'submit_' + str(sid))(n_estimators, eta, nrows)


def main():
    config.setup_logger()

    fire.Fire({
        'submit': submit
    })
