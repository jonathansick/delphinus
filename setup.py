#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import sys

import glob
import os
from setuptools import setup

# A dirty hack to get around some early import/configurations ambiguities
if sys.version_info[0] >= 3:
    import builtins
else:
    import __builtin__ as builtins
builtins._ASTROPY_SETUP_ = True

from astropy_helpers.setup_helpers import (
    register_commands, adjust_compiler, get_debug_option)
from astropy_helpers.git_helpers import get_git_devstr
from astropy_helpers.version_helpers import generate_version_py

# Set affiliated package-specific settings
PACKAGENAME = 'delphinus'
DESCRIPTION = 'Python interface to the DOLPHOT photometry tool.'
LONG_DESCRIPTION = ''
AUTHOR = 'Jonathan Sick'
AUTHOR_EMAIL = ''
LICENSE = 'BSD'
URL = 'http://jonathansick.ca'

# VERSION should be PEP386 compatible (http://www.python.org/dev/peps/pep-0386)
VERSION = '0.2.dev'

# Indicates if this version is a release version
RELEASE = 'dev' not in VERSION

if not RELEASE:
    VERSION += get_git_devstr(False)

# Populate the dict of setup command overrides; this should be done before
# invoking any other functionality from distutils since it can potentially
# modify distutils' behavior.
cmdclassd = register_commands(PACKAGENAME, VERSION, RELEASE)

# Adjust the compiler in case the default on this platform is to use a
# broken one.
adjust_compiler(PACKAGENAME)

# Freeze build information in version.py
generate_version_py(PACKAGENAME, VERSION, RELEASE,
                    get_debug_option('delphinus'))

# Use the find_packages tool to locate all packages and modules
packagenames = ['delphinus']

# Treat everything in scripts except README.rst as a script to be installed
scripts = [fname for fname in glob.glob(os.path.join('scripts', '*'))
           if os.path.basename(fname) != 'README.rst']

# Additional C extensions that are not Cython-based should be added here.
extensions = []

# A dictionary to keep track of all package data to install
package_data = {PACKAGENAME: ['data/*']}

# A dictionary to keep track of extra packagedir mappings
package_dirs = {}


setup(name=PACKAGENAME,
      version=VERSION,
      description=DESCRIPTION,
      packages=packagenames,
      package_data=package_data,
      package_dir=package_dirs,
      ext_modules=extensions,
      scripts=scripts,
      requires=['astropy'],
      install_requires=['astropy'],
      provides=[PACKAGENAME],
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      url=URL,
      long_description=LONG_DESCRIPTION,
      cmdclass=cmdclassd,
      zip_safe=False,
      use_2to3=True
      )
