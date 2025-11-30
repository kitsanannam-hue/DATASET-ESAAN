
"""Setup script for AudioCraft AI Agents."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="audiocraft-ai-agents",
    version="0.1.0",
    author="Your Name",
    description="AI Agents for AudioCraft audio generation models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "hydra-core>=1.3.0",
        "numpy>=1.24.0",
        "librosa>=0.10.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
