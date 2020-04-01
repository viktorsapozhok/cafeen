from setuptools import setup

SHORT_DESCRIPTION = ''

DEPENDENCIES = [
    'click==7.1.1',
    'colorlog',
    'eli5==0.10.1',
    'optuna==1.2.0',
    'pandas==1.0.3',
    'scikit-learn>=0.22.0',
    'scipy==1.4.1',
    'statsmodels>=0.11.1'
]

ENTRY_POINTS = {
    'console_scripts': [
        'cafeen=cafeen.cli:cafeen'
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
