from setuptools import setup, find_packages

setup(
    name="or_algorithms",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "PuLP==2.9.0",
        "numpy>=1.14.5",
        "networkx==3.4.2",
        "sortedcontainers==2.4.0",
        "scipy==1.15.2",
    ],
    extras_require={
        "interactive": ["jupyter", "matplotlib"],
        "dev": ["pytest", "flake8", "mypy", "black"],
    },
)
