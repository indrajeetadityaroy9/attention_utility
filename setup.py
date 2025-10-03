from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bert-attention-toolkit",
    version="1.0.0",
    author="",
    description="Unified CLI tool for extracting and analyzing transformer attention patterns",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "pyarrow>=8.0.0",
        "scipy>=1.7.0",
    ],
    scripts=[
        "bert-attention.py",
    ],
)
