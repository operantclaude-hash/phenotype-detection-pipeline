from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="phenotype-detection-pipeline",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Artifact-free phenotype detection using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/phenotype-detection-pipeline",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pytorch-lightning>=2.0.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "h5py>=3.8.0",
        "Pillow>=9.5.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.65.0",
    ],
)
