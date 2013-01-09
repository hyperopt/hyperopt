#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" distribute- and pip-enabled setup.py """

import logging
import os
import re

# ----- overrides -----

# set these to anything but None to override the automatic defaults
packages = None
package_name = None
package_data = None
scripts = None
requirements_file = None
requirements = None
dependency_links = None
# ---------------------


# ----- control flags -----

# fallback to setuptools if distribute isn't found
setup_tools_fallback = True

# don't include subdir named 'tests' in package_data
skip_tests = False

# print some extra debugging info
debug = True

# -------------------------

if debug: logging.basicConfig(level=logging.DEBUG)
# distribute import and testing
try:
    import distribute_setup
    distribute_setup.use_setuptools()
    logging.debug("distribute_setup.py imported and used")
except ImportError:
    # fallback to setuptools?
    # distribute_setup.py was not in this directory
    if not (setup_tools_fallback):
        import setuptools
        if not (hasattr(setuptools,'_distribute') and \
                setuptools._distribute):
            raise ImportError("distribute was not found and fallback to setuptools was not allowed")
        else:
            logging.debug("distribute_setup.py not found, defaulted to system distribute")
    else:
        logging.debug("distribute_setup.py not found, defaulting to system setuptools")

import setuptools

def find_scripts():
    return [s for s in setuptools.findall('scripts/') if os.path.splitext(s)[1] != '.pyc']

def package_to_path(package):
    """
    Convert a package (as found by setuptools.find_packages)
    e.g. "foo.bar" to usable path
    e.g. "foo/bar"

    No idea if this works on windows
    """
    return package.replace('.','/')

def find_subdirectories(package):
    """
    Get the subdirectories within a package
    This will include resources (non-submodules) and submodules
    """
    try:
        subdirectories = os.walk(package_to_path(package)).next()[1]
    except StopIteration:
        subdirectories = []
    return subdirectories

def subdir_findall(dir, subdir):
    """
    Find all files in a subdirectory and return paths relative to dir

    This is similar to (and uses) setuptools.findall
    However, the paths returned are in the form needed for package_data
    """
    strip_n = len(dir.split('/'))
    path = '/'.join((dir, subdir))
    return ['/'.join(s.split('/')[strip_n:]) for s in setuptools.findall(path)]

def find_package_data(packages):
    """
    For a list of packages, find the package_data

    This function scans the subdirectories of a package and considers all
    non-submodule subdirectories as resources, including them in
    the package_data

    Returns a dictionary suitable for setup(package_data=<result>)
    """
    package_data = {}
    for package in packages:
        package_data[package] = []
        for subdir in find_subdirectories(package):
            if '.'.join((package, subdir)) in packages: # skip submodules
                logging.debug("skipping submodule %s/%s" % (package, subdir))
                continue
            if skip_tests and (subdir == 'tests'): # skip tests
                logging.debug("skipping tests %s/%s" % (package, subdir))
                continue
            package_data[package] += subdir_findall(package_to_path(package), subdir)
    return package_data

def parse_requirements(file_name):
    """
    from:
        http://cburgmer.posterous.com/pip-requirementstxt-and-setuppy
    """
    requirements = []
    with open(file_name, 'r') as f:
        for line in f:
            if re.match(r'(\s*#)|(\s*$)', line): continue
            if re.match(r'\s*-e\s+', line):
                requirements.append(re.sub(r'\s*-e\s+.*#egg=(.*)$',\
                        r'\1', line).strip())
            elif re.match(r'\s*-f\s+', line):
                pass
            else:
                requirements.append(line.strip())
    return requirements

def parse_dependency_links(file_name):
    """
    from:
        http://cburgmer.posterous.com/pip-requirementstxt-and-setuppy
    """
    dependency_links = []
    with open(file_name) as f:
        for line in f:
            if re.match(r'\s*-[ef]\s+', line):
                dependency_links.append(re.sub(r'\s*-[ef]\s+',\
                        '', line))
    return dependency_links

# ----------- Override defaults here ----------------
if packages is None: packages = setuptools.find_packages()

if len(packages) == 0: raise Exception("No valid packages found")

if package_name is None: package_name = packages[0]

if package_data is None: package_data = find_package_data(packages)

if scripts is None: scripts = find_scripts()

if requirements_file is None:
    requirements_file = 'requirements.txt'

if os.path.exists(requirements_file):
    if requirements is None:
        requirements = parse_requirements(requirements_file)
    if dependency_links is None:
        dependency_links = parse_dependency_links(requirements_file)
else:
    if requirements is None:
        requirements = []
    if dependency_links is None:
        dependency_links = []

if debug:
    logging.debug("Module name: %s" % package_name)
    for package in packages:
        logging.debug("Package: %s" % package)
        logging.debug("\tData: %s" % str(package_data[package]))
    logging.debug("Scripts:")
    for script in scripts:
        logging.debug("\tScript: %s" % script)
    logging.debug("Requirements:")
    for req in requirements:
        logging.debug("\t%s" % req)
    logging.debug("Dependency links:")
    for dl in dependency_links:
        logging.debug("\t%s" % dl)

# -- HACK to make sure the hard-coded requirements stay in sync with
#    requirements.txt which is *not* included in releases.

_hard_code_requirements = ['numpy', 'scipy', 'nose', 'pymongo', 'networkx']
if requirements:
    assert set(requirements) == set(_hard_code_requirements), (
        requirements, _hard_code_requirements)

setuptools.setup(
    name = package_name,
    version = '0.0.1',
    packages = packages,
    scripts = scripts,
    url = 'http://jaberg.github.com/hyperopt/',
    author = 'James Bergstra',
    author_email = 'anon@anon.com',
    description = 'Distributed Asynchronous Hyperparameter Optimization',
    long_description = open('README.txt').read(),
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Environment :: Console',
        'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
    ],
    platforms = ['Linux', 'OS-X', 'Windows'],
    license = 'BSD',
    keywords = 'Bayesian optimization hyperparameter model selection',
    package_data = package_data,
    include_package_data = True,
    install_requires = _hard_code_requirements,
    #dependency_links = dependency_links # -- what are these?
)
