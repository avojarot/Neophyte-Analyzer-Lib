"""
Setup module for install lib
"""
import os
from os import path
from typing import List

from setuptools import setup

MODULE_NAME = 'neophyte_analyzer'
LIB_NAME = 'neophyte_analyzer'
__version__ = '0.0.1'

this_directory = path.abspath(path.dirname(__file__))
try:
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        LONG_DESCRIPTION = f.read()
except FileNotFoundError:
    LONG_DESCRIPTION = 'neophyte_analyzer for analyzing facial expressions'


def get_packages() -> List[str]:
    """
    Help method
    :return: List[str] path to files and folders library
    """
    ignore = ['__pycache__']

    list_sub_folders_with_paths = [x[0].replace(os.sep, '.')
                                   for x in os.walk(MODULE_NAME)
                                   if x[0].split(os.sep)[-1] not in ignore]
    return list_sub_folders_with_paths


setup(
    name=LIB_NAME,
    version=__version__,
    description='Library for analyzing facial expressions',
    author='M.Kizitskyi',
    author_email='maksym.kizitskyi@nure.ua',
    packages=get_packages(),
    keywords=['pip', MODULE_NAME],
    packages_dir={MODULE_NAME: MODULE_NAME},
    long_description=open('README.md').read(),
    install_requires=["cv2",
                      "numpy~=1.21.1",
                      'mediapipe',
                      'tensorflow'
                      ]
)
