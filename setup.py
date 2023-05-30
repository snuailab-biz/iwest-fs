from setuptools import setup, find_packages
from pathlib import Path
from setuptools.command.install import install
import subprocess
import pkg_resources as pkg

FILE = Path(__file__).resolve()
PARENT = FILE.parent  # root directory
README = (PARENT / 'README.md').read_text(encoding='utf-8')
REQUIREMENTS = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements((PARENT / 'requirements.txt').read_text())]

class CustomInstall(install):
    def run(self):
        # 기본 설치 작업 실행
        install.run(self)
        
        # make.sh 스크립트 실행
        subprocess.call(['pwd'])
        # subprocess.call(['sh', 'make.sh'])
        # subprocess.call(['sh', 'make.sh'])

setup(
    name = 'iwestfs',
    version = '0.0.2',
    description='setup iwest fire & smoke detector install.',
    url='https://github.com/snuailab-biz/iwest-fs.git',
    author='leejj',
    author_email="leejj@snuailab.ai",
    license='snuailab',
    packages=find_packages(),
    include_package_data=True,
    install_requires=REQUIREMENTS,
    cmdclass={
        'install': CustomInstall,
    },
)