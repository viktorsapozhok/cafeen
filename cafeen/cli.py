import click

from . import config, clients


@click.group()
def cafeen():
    """Categorical Feature Encoding Challenge"""

    pass


@cafeen.command()
def submit():
    clients.predict(n_valid_rows=0)


@cafeen.command()
@click.option('--rows', default=0, type=int,
              help='Number of rows in validation set')
def validate(rows):
    config.setup_logger()

    clients.validate(n_valid_rows=rows)


if __name__ == '__main__':
    cafeen()
