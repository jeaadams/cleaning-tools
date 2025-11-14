from setuptools import setup, find_packages

setup(
    name="cleaning-tools",
    version="0.1.0",
    description="General ALMA data cleaning and visualization tools",
    author="Your Name",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "matplotlib",
        "astropy",
        "scipy",
        "bettermoments",
    ],
    python_requires=">=3.8",
)
