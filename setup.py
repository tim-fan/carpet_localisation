# -*- coding: utf-8 -*-
"""setup.py"""

import os
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand


class Tox(TestCommand):
    user_options = [('tox-args=', 'a', 'Arguments to pass to tox')]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.tox_args = None

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import tox
        import shlex
        if self.tox_args:
            errno = tox.cmdline(args=shlex.split(self.tox_args))
        else:
            errno = tox.cmdline(self.test_args)
        sys.exit(errno)


def read_content(filepath):
    with open(filepath) as fobj:
        return fobj.read()


classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved ::"
    " GNU General Public License v3 or later (GPLv3+)",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: Implementation :: CPython",
    "Programming Language :: Python :: Implementation :: PyPy",
]

long_description = (read_content("README.md") +
                    read_content(os.path.join("docs/source", "CHANGELOG.rst")))

requires = [
    'docopt',
    'matplotlib',
    'numpy',
    'opencv-python',
    'pfilter @ git+https://github.com/johnhw/pfilter.git',
    'scipy',
    'setuptools',
]

extras_require = {
    'reST': ['Sphinx'],
}
if os.environ.get('READTHEDOCS', None):
    extras_require['reST'].append('recommonmark')

setup(
    name='cbl_particle_filter',
    version='0.1.0',
    description='##### ToDo: Rewrite me #####',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    author='Tim Fanselow',
    author_email='author@email.com',
    url='https://github.com/tim-fan/cbl_particle_filter',
    classifiers=classifiers,
    packages=['cbl_particle_filter'],
    data_files=[],
    install_requires=requires,
    include_package_data=True,
    extras_require=extras_require,
    entry_points={
        'console_scripts': [
            'carpet_map_csv_to_png=cbl_particle_filter.bin.carpet_map_csv_to_png:main'
        ],
    },
    tests_require=['tox'],
    cmdclass={'test': Tox},
)
