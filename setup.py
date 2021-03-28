import codecs
import os

from setuptools import find_packages, setup


def read(fname):
    return codecs.open(os.path.join(base_dir, fname), encoding="utf-8").read()


# https://packaging.python.org/single_source_version/
base_dir = os.path.abspath(os.path.dirname(__file__))
about = {}
with open(os.path.join(base_dir, "gyptis", "__about__.py"), "rb") as f:
    exec(f.read(), about)


required = [
    "scipy",
    "dolfin-adjoint",
    "meshio",
    "gmsh",
    "h5py",
    "nlopt",
    "numpy",
    "simpy",
]

_classifiers = [
    about["__status__"],
    about["__license__"],
    about["__operating_system__"],
]
_classifiers += about["__programming_language__"]
_classifiers += about["__topic__"]

setup(
    name="gyptis",
    version=about["__version__"],
    author=about["__author__"],
    author_email=about["__author_email__"],
    packages=find_packages(),
    description=about["__description__"],
    long_description=read("README.rst"),
    long_description_content_type="text/x-rst",
    url=about["__website__"],
    project_urls={"Documentation": about["__website__"]},
    license=about["__license__"],
    platforms="any",
    include_package_data=True,
    install_requires=required,
    extras_require={},
    classifiers=_classifiers,
)
