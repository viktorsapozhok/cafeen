from setuptools import setup

SHORT_DESCRIPTION = ''

DEPENDENCIES = [
    'colorlog',
    'eli5==0.10.1',
    'fire==0.2.1',
    'lightgbm==2.3.1',
    'optuna==1.0.0',
    'pandas==0.25.3',
    'scikit-learn==0.22.1',
    'tqdm'
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
