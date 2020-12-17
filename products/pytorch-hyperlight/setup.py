from setuptools import setup, find_namespace_packages
from pip._internal.req import parse_requirements

# flake8: noqa

NAME = "pytorch_hyperlight"

NAMESPACE = "pytorch_hyperlight"
SERVICE = "base"
NAME = "{}.{}".format(NAMESPACE, SERVICE)

install_reqs = parse_requirements("requirements.txt", "req_install_hack")
reqs = [str(ir.requirement) for ir in install_reqs]

setup(
    name=NAME,
    version="0.1-DEVELOP",
    description="PyTorch HyperLight",
    long_description="",
    author="Peter Gagarinov",
    author_email="Peter Gagarinov <pgagarinov@gmail.com>",
    url="https://github.com/pgagarinov/pytorch-hyperlight.git",
    zip_safe=True,
    packages=find_namespace_packages(include=[NAMESPACE + ".*"]),
    install_requires=reqs,
    entry_points={},
    dependency_links=[],
)
