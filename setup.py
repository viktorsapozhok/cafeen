from setuptools import setup

SHORT_DESCRIPTION = ''

DEPENDENCIES = [
    'click==7.1.1',
    'colorlog',
    'optuna==1.2.0',
    'pandas==1.0.3',
    'scikit-learn>=0.22.0',
]

ENTRY_POINTS = {
    'console_scripts': [
        'cafeen=cafeen.cli:main'
    ]
}

VERSION = '0.0.1'
URL = 'https://github.com/viktorsapozhok/cafeen'

setup(
    name='cafeen',
    version=VERSION,
    description=SHORT_DESCRIPTION,
    long_description=SHORT_DESCRIPTION,
    url=URL,
    author='viktorsapozhok',
    license='MIT License',
    packages=['cafeen'],
    install_requires=DEPENDENCIES,
    entry_points=ENTRY_POINTS
)
