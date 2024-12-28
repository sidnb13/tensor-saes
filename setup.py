from setuptools import find_packages, setup

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="sae",
    version="0.1.0",
    description="Sparse autoencoders",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(include=["sae*"]),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "sae=sae.__main__:run",
        ],
    },
    license="MIT",
    keywords=["interpretability", "explainable-ai"],
)
