from setuptools import setup, find_packages

setup(
    name="ember",
    version="1.3.10",
    description="Basic utils and base classes for training and evaluating models in PyTorch",
    author="Georgios Chochlakis",
    author_email="chochlak@usc.edu",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "scikit-learn",
        "legm @ git+https://github.com/gchochla/legm.git@main",
        "accelerate",
        "gridparse @ git+https://github.com/gchochla/gridparse.git@main",
    ],
    extras_require={"dev": ["black", "pytest"]},
)
