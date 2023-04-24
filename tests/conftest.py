"""Sets root directory for repo."""
import os
import sys
from os.path import abspath, dirname

root_dir = dirname(dirname(abspath(__file__)))
sys.path.append(root_dir)
