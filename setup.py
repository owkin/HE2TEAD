from setuptools import setup, find_packages

setup(
    name='he2tead',
    version=0.1,
    author='OWKIN',
    description='Code repository for HE2TEAD',
    packages=find_packages(),
    install_requires=[
      'tqdm',
      'fire',
      'loguru',
      'pandas>=1.0',
      'numpy',
      'scikit-learn',
      'torch',
      'seaborn',
      'openslide-python',
      'mlflow',
    ],
    long_description=open('README.md').read()
)
