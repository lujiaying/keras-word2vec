from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='keras-word2vec',
    version='0.1.0',

    description='Word2vec implement in Keras.',
    long_description=long_description,

    url='https://github.com/lujiaying/keras-word2vec',
    author='Jiaying Lu',
    author_email='lujiaying93@foxmail.com',
    license='MIT',

    install_requires=[
        'keras>=2.0.6',
        ],

    extras_require={
        'test': ['nose'],
    },


    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules'
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
    ],

    keywords='word2vec machineLearning deepLearing',

    packages=find_packages(),
)
