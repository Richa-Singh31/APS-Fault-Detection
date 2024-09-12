from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str = 'requirements.txt')->List[str]:
    """
    This function reads the requirements.txt file and returns a list of required packages.
    """
    with open(file_path, 'r') as file:
        requirements = file.readlines()
        requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('-e')]
    return requirements

setup(
    name="sensor",
    version="0.0.1",
    author_email="richarsingh31@gmail.com",
    packages= find_packages(),
    install_requires = get_requirements(),
)
