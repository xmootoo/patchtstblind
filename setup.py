from setuptools import setup, find_packages

# Read dependencies from requirements.txt
required = []
with open('requirements.txt') as f:
	required = f.read().splitlines()

setup(name='patchtstblind',
	  version='0.0.1',
	  packages=find_packages(),
	  description='Time Series Joint-Embedding Predictive Architecture (JEPA)',
      # requires=required,
	  author='Xavier Mootoo, Luca Vivona',
	  author_email='xmootoo@my.yorku.ca, luca01@my.yorku.ca',
	  # license='MIT',install_requires=required,classifiers=['Programming Language :: Python :: 3','License :: OSI Approved :: MIT License','Operating System :: OS Independent',],
	  python_requires='>=3.10')
