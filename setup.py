"""This file is for setup information and version control used."""
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as f:
    readme_text = f.read()

with open('LICENSE') as f:
    license_text = f.read()

setup(
    project_name='kaggle-house-prices',
    project_version='0.2.1',
    description='In this project, we use machine learning to predict the regression data.',
    long_description=readme_text,
    author='Simon Liu',
    url='https://github.com/LiuYuWei/kaggle-house-prices',
    license=license_text,
    packages=find_packages(exclude=('tests'))
)
