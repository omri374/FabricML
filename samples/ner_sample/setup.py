from setuptools import find_packages, setup

setup(
    name="ner_sample",
    packages=find_packages(),
    version="0.1.0",
    description="Example of using the ML experimentation framework to solve a Named Entity Recognition problem",
    author="Omri Mendels",
    license="MIT",
    entry_points={
        "console_scripts": ["generate_notebook=generate_notebook:generate_notebook"],
    },
)
