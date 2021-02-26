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

import os
from io import open

from setuptools import find_packages, setup

CURR_DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(CURR_DIR, "README.md"), encoding="utf-8") as file_open:
    LONG_DESCRIPTION = file_open.read()

setup(
    name='synthetic-data',
    version='1.0.0',
    maintainer='Brian Barr',
    maintainer_email='brian.barr@capitalone.com',
    license='Apache License 2.0',
    description='Generates complex, nonlinear datasets for use \
        with deep learning/black box models',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://github.com/capitalone/synthetic-data',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "copulas > 0.3.2",
        "shap > 0.34.0",
        "sympy > 1.6.1",
        "tensorflow > 2.3.0",
        "qii-tool > 0.1.2"
    ],
    python_requires=">=3.6"
)
