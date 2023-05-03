# SPDX-Copyright: Copyright (c) Capital One Services, LLC
# SPDX-License-Identifier: Apache-2.0
# Copyright 2018 Capital One Services, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages


def parse_requirements(filename):
    """Loads requirements file and outputs an array of dependencies"""
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith('#')]


with open('README.md', 'r') as readme:
    long_description = readme.read()

setup(
    name='synthetic-data',
    version='1.2.0',
    maintainer='Brian Barr',
    maintainer_email='brian.barr@capitalone.com',
    license='Apache License 2.0',
    description='Generates complex, nonlinear datasets for use \
        with deep learning/black box models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=parse_requirements('requirements.txt'),
    url='https://github.com/capitalone/synthetic-data',
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8"
)
