from setuptools import setup, find_packages
from pathlib import Path

import pkg_resources as pkg
FILE = Path(__file__).resolve()
PARENT = FILE.parent  # root directory
README = (PARENT / 'README.md').read_text(encoding='utf-8')
REQUIREMENTS = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements((PARENT / 'requirements.txt').read_text())]

setup(
    name = 'iwestfs',
    version = '0.0.1',
    description='setup iwest fire & smoke detector install.',
    url='https://github.com/snuailab-biz/iwest-fs.git',
    author='leejj',
    author_email="leejj@snuailab.ai",
    license='snuailab',
    packages=find_packages(),
    include_package_data=True,
    install_requires=REQUIREMENTS
)