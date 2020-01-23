from setuptools import setup

SHORT_DESCRIPTION = ''

DEPENDENCIES = [
    'colorlog',
    'fire==0.2.1',
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
