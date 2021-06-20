from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


__version__ = ""
with open('VERSION') as version_file:
    __version__ = version_file.read().strip()


setup(
    name='mlfabric',
    packages=find_packages(),
    version=__version__,
    description='Experimentation framework for data science teams',
    long_description=long_description,
    license='MIT'

)
