import setuptools

requirements = open("requirements.txt").read().split()
VERSION = "0.0.1"
setuptools.setup(
    name="effective_transformers",
    version=VERSION,
    author="HawkeOni",
    author_email="iliyadimov@icloud.com",
    description="Effective transformer examples benchmarked on listops.",
    install_requires=requirements,
    python_requires=">=3.6.1"
)