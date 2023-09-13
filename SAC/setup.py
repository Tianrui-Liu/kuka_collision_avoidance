from setuptools import setup

setup(
    name="path_planning_kuka",
    version="0.0.1",
    py_modules=['path_planning_kuka'],
    packages=['path_planning_kuka'],
    package_dir={'path_planning_kuka': 'path_planning/envs'},
    install_requires=["gym", "torch", "pybullet"],
)
