#!C:\Users\pypy2\AppData\Local\Programs\Python\Python311\python.exe

from __future__ import print_function
from setuptools import setup, find_packages
import sys


setup(
    name="llmkits",
    version="0.1.0",
    author="Wingoal",  # 作者名字
    author_email="panwingoal@gmail.com",
    description="Some useful helper functions for LLM models analysis",
    license="MIT",
    url="https://github.com/wingoalpan/llmkits.git",  # github地址或其他地址
    # packages=find_packages(),
    packages=['llmkits'],
    package_dir={'llmkits': '.'},
    # package_data={'config': ['config/*.json']},
    include_package_data=False,
    classifiers=[
        "Environment :: Windows Environment",
        'Intended Audience :: AI LLM developer',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: Chinese',
        'Operating System :: Microsoft',
        'Operating System :: POSIX',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3.11',
    ],
    install_requires=[
        'torch',
        'transformers',
        'safetensors',
        'gguf',
        'pandas',
        'openpyxl'
    ],
    zip_safe=True,
)
