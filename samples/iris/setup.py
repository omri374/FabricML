from setuptools import find_packages, setup

setup(
    name="iris",
    packages=find_packages(),
    version="0.1.0",
    description="A short description of the project.",
    author="Omri",
    license="MIT",
    install_requires=["numpy", "sklearn", "pandas"],
    entry_points={
        "console_scripts": ["generate_notebook=generate_notebook:generate_notebook"],
    },
)
