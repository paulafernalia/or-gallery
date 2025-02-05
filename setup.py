from setuptools import setup, find_packages

setup(
    name="or_algorithms",
    version="0.1",
    packages=find_packages(where="src"),  # Tell setuptools to look inside 'src'
    package_dir={"": "src"},  # Maps root package names to 'src/'
    install_requires=[
        'PuLP==2.9.0',
        'numpy>=1.14.5'
    ],
    extras_require={
        'interactive': ['jupyter'],
        'dev': ['pytest', 'flake8'],  # Add test dependencies under "dev"
    },
)