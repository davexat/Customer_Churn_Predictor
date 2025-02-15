import os

from setuptools import setup, find_packages

def readme() -> str:
    """Utility function to read the README.md.

    Used for the `long_description`. It's nice, because now
    1) we have a top level README file and
    2) it's easier to type in the README file than to put a raw string in below.

    Args:
        nothing

    Returns:
        String of readed README.md file.
    """
    return open(os.path.join(os.path.dirname(__file__), 'README.md')).read()

setup(
    name='churn_predictor',
    version='0.1.0',
    author='David Sandoval',
    author_email='daelsand@espol.edu.ec',
    description='A customer churn prediction project that leverages machine learning to assess the risk of customer abandonment, while providing an interactive dashboard to visualize key factors and retention metrics.',
    python_requires='>=3',
    url='',
    packages=find_packages(),
    long_description=readme(),
)