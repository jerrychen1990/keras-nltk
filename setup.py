# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     setup
   Description :
   Author :       chenhao
   date：          2019-06-26
-------------------------------------------------
   Change Activity:
                   2019-06-26:
-------------------------------------------------
"""

from setuptools import setup, find_packages

setup(
    name='eigen_nltk',
    version='0.0.1',
    packages=find_packages(exclude=['tests*']),
    package_dir={"": "."},
    url='git@git.aipp.io:chenhao/person_text_extract.git',
    license='MIT',
    author='Chen Hao',
    author_email='chenhao@aidigger.com',
    zip_safe=True,
    description='extract person schema from text',
)
