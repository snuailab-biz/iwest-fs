from setuptools import setup, find_packages
from pathlib import Path

import pkg_resources as pkg
FILE = Path(__file__).resolve()
PARENT = FILE.parent  # root directory
README = (PARENT / 'README.md').read_text(encoding='utf-8')
REQUIREMENTS = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements((PARENT / 'requirements.txt').read_text())]

setup(
    name = 'rfai',
    version = '0.0.4',
    description='setup rfai install.',
    url='https://github.com/RearFold/RearFold-AI.git',
    author='kaejong2',
    author_email="kaejong2@gmail.com",
    license='ljj',
    packages=find_packages(),
    include_package_data=True,
    install_requires=REQUIREMENTS
)