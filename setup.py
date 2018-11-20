import os
import re
import codecs

from setuptools import setup, find_packages
from os import path
from io import open

cwd = os.path.abspath(os.path.dirname(__file__))

def read(filename):
    with codecs.open(os.path.join(cwd, filename), 'rb', 'utf-8') as h:
        return h.read()

metadata = read(os.path.join('rlcc', '__init__.py'))

def extract_metaitem(meta):
    # Based on the python twitter setup
    meta_match = re.search(r"""^__{meta}__\s+=\s+['\"]([^'\"]*)['\"]""".format(meta=meta),
                           metadata, re.MULTILINE)
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError('Unable to find __{meta}__ string.'.format(meta=meta))

long_description = read('README.md')

setup(
    name='rlcc',
    version=extract_metaitem('version'),
    description=extract_metaitem('description'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=extract_metaitem('url'),
    author=extract_metaitem('author'),
    author_email=extract_metaitem('email'),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='reinforcement-learning rl',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=['torch'],
    project_urls={
        'Bug Reports': 'https://github.com/pjordan/rlcc/issues',
        'Source': 'https://github.com/pjordan/rlcc/',
    },
)