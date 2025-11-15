from setuptools import setup, find_packages

setup(
    name="cleaning-tools",
    version="0.1.0",
    description="General ALMA data cleaning and visualization tools",
    author="Jea Adams Redai",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "matplotlib",
        "astropy",
        "scipy",
        "bettermoments",
        "ipywidgets",
    ],
    extras_require={
        "interactive": ["ipympl"],  # For faster interactive performance
    },
    python_requires=">=3.8",
)
