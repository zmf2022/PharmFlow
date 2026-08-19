from setuptools import find_namespace_packages, setup

setup(
    packages=find_namespace_packages(
        include=("pharm_flow*",)
    ),
    include_package_data=True,
)
