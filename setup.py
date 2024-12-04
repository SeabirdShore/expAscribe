from setuptools import setup, find_packages

setup(
    name='expAscribe',
    version='1.2.9',
    author='$3@6iRd5h0r3',
    description='ExpAscribe: a causal inference framework for quantitative experiment ascription and its derivative process. Documentation: https://seabirdshore.github.io/EAdocs/',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    install_requires=[
        'causal_learn==0.1.3.8',
        'dagma==1.1.1',
        'dowhy==0.11.1',
        'networkx>=3.2.1',
        'bayesian-optimization==1.4.2',
        'scikit_learn>=1.5.1',
        'scipy>=1.11.4',
        'statsmodels>=0.14.2',
        'matplotlib>=3.7.0',
        'numpy>=1.23.0,<1.26.4',
        'pandas>=1.5.3',
    ],
)
