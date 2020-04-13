from setuptools import setup

SHORT_DESCRIPTION = ''

DEPENDENCIES = [
    'click==7.1.1',
    'colorlog',
    'numpy',
    'pandas==1.0.3',
    'scikit-learn>=0.22.0',
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
