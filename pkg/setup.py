from setuptools import find_packages, setup

REQUIRED_PACKAGES = [
    "graspologic",
]

setup(
    name="pkg",
    packages=find_packages(),
    version="0.1.0",
    description="Local package for bilateral matching paper",
    author="Neurodata",
    license="MIT",
    install_requires=REQUIRED_PACKAGES,
    dependency_links=[],
)
