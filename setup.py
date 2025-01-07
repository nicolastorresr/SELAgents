"""
Setup script for SELAgents package.
"""

from setuptools import setup, find_packages
import os
import codecs
from pathlib import Path

# Get the long description from the README file
def read_long_description():
    here = Path(__file__).parent.resolve()
    with codecs.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        return f.read()

# Get version from __init__.py
def get_version():
    here = Path(__file__).parent.resolve()
    init_path = here / 'selagents' / '__init__.py'
    with codecs.open(init_path, encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip("'").strip('"')
    raise RuntimeError('Version not found')

# Dependencies
REQUIRED_PACKAGES = [
    'numpy>=1.21.0',
    'torch>=1.9.0',
    'pandas>=1.3.0',
    'networkx>=2.6.0',
    'matplotlib>=3.4.0',
    'seaborn>=0.11.0',
    'scipy>=1.7.0',
    'scikit-learn>=0.24.0'
]

# Development dependencies
DEVELOPMENT_PACKAGES = [
    'pytest>=6.2.0',
    'pytest-cov>=2.12.0',
    'black>=21.5b2',
    'isort>=5.9.0',
    'flake8>=3.9.0',
    'mypy>=0.910',
    'sphinx>=4.0.0',
    'sphinx-rtd-theme>=0.5.0',
    'twine>=3.4.0'
]

setup(
    name='selagents',
    version=get_version(),
    description='A framework for social-emotional learning in artificial agents',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    author='Nicolas Torres',
    author_email='nicolas.torresr@usm.cl',
    url='https://github.com/nicolastorresr/SELAgents',
    license='MIT',
    
    # Package info
    packages=find_packages(include=['selagents', 'selagents.*']),
    python_requires='>=3.8',
    zip_safe=False,
    
    # Dependencies
    install_requires=REQUIRED_PACKAGES,
    extras_require={
        'dev': DEVELOPMENT_PACKAGES,
        'docs': [
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=0.5.0',
            'sphinx-autodoc-typehints>=1.12.0'
        ],
        'test': [
            'pytest>=6.2.0',
            'pytest-cov>=2.12.0'
        ]
    },
    
    # Classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Operating System :: OS Independent',
    ],
    
    # Keywords
    keywords=[
        'artificial intelligence',
        'social learning',
        'emotional intelligence',
        'reinforcement learning',
        'theory of mind',
        'multi-agent systems',
        'social networks',
        'emotional processing'
    ],
    
    # Project URLs
    project_urls={
        'Bug Reports': 'https://github.com/nicolastorresr/selagents/issues',
        'Documentation': 'https://selagents.readthedocs.io/',
        'Source': 'https://github.com/nicolastorresr/selagents',
    },
    
    # Package data
    package_data={
        'selagents': [
            'data/*.json',
            'data/*.pkl',
            'config/*.yaml',
        ],
    },
    include_package_data=True,
    
    # Entry points
    entry_points={
        'console_scripts': [
            'selagents=selagents.cli:main',
        ],
    }
)
