import os
from setuptools import find_packages, setup

NAME = "gpstardist"
DESCRIPTION = "Gunpowder node for stardist computation"
URL = "https://github.com/saalfeldlab/gunpowder-stardist"
EMAIL = "heinrichl@janelia.hhmi.org"
AUTHOR = "Larissa Heinrich"
REQUIRES_PYTHON = ">=3.6"
VERSION = "0.1.dev1"

REQUIRED = [
    "tensorflow_gpu<1.15",
    "numpy",
    "stardist",
    "gunpowder"
]

EXTRAS = {
}

DEPENDENCY_LINKS = [
]

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), "r") as f:
    LONG_DESCRIPTION = "\n" + f.read()

with open(os.path.join(here, "ACKNOWLEDGMENTS"), "r") as f:
    LONG_DESCRIPTION += "\n\n"
    LONG_DESCRIPTION += f.read()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    ext_modules=[],
    entry_points={},
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    dependency_links=DEPENDENCY_LINKS,
    include_package_data=True,
    license="BSD-2-Clause",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
