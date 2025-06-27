"""
OpenNeighbor: Production-Grade Neighborhood-Aware Recommendation System
Author: Nik Jois <nikjois@llamasearch.ai>
"""
from setuptools import setup, find_packages
import os

# Read version from __init__.py
def get_version():
    init_py_path = os.path.join('openneighbor', '__init__.py')
    if os.path.exists(init_py_path):
        with open(init_py_path) as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"\'')
    return '1.0.0'

# Read README for long description
def get_long_description():
    if os.path.exists('README.md'):
        with open('README.md', 'r', encoding='utf-8') as fh:
            return fh.read()
    return "OpenNeighbor: Production-Grade Neighborhood-Aware Recommendation System"

# Read requirements
def get_requirements():
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return [
        'torch>=2.1.0',
        'torch-geometric>=2.4.0',
        'transformers>=4.35.0',
        'sentence-transformers>=2.2.2',
        'numpy>=1.24.0',
        'pandas>=2.1.0',
        'scikit-learn>=1.3.0',
        'fastapi>=0.104.0',
        'uvicorn[standard]>=0.24.0',
        'pydantic>=2.4.0',
        'click>=8.1.0',
        'rich>=13.6.0',
        'typer>=0.9.0',
        'pyyaml>=6.0.1',
        'networkx>=3.2',
        'geopy>=2.4.0',
        'structlog>=23.1.0',
        'prometheus-client>=0.18.0',
    ]

setup(
    name='openneighbor',
    version=get_version(),
    author='Nik Jois',
    author_email='nikjois@llamasearch.ai',
    description='Production-Grade Neighborhood-Aware Recommendation System',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/nikjois/openneighbor',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8',
    install_requires=get_requirements(),
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'pytest-asyncio>=0.21.0',
            'black>=23.0.0',
            'isort>=5.12.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
            'pre-commit>=3.0.0',
            'jupyter>=1.0.0',
        ],
        'docs': [
            'sphinx>=6.0.0',
            'sphinx-rtd-theme>=1.2.0',
            'myst-parser>=1.0.0',
        ],
        'benchmarks': [
            'memory-profiler>=0.60.0',
            'py-spy>=0.3.0',
            'locust>=2.14.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'openneighbor=openneighbor.ui.cli:cli',
        ],
    },
    include_package_data=True,
    package_data={
        'openneighbor': [
            'ui/web/templates/**/*',
            'ui/web/static/**/*',
            'deployment/**/*',
        ],
    },
    zip_safe=False,
) 