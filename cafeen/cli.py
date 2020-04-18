import click

from . import config, steps


@click.group()
def cafeen():
    """Categorical Feature Encoding Challenge II"""

    pass


@cafeen.command()
def submit():
    """Make a submission.
    """

    pass


@cafeen.command()
def validate():
    """Cross-validation.
    """

    config.setup_logger()
    steps.cross_val()


if __name__ == '__main__':
    cafeen()
