from setuptools import setup, find_packages

setup(
    name="nimble",
    version="0.1.0",
    description="Image-to-Vector Sketch Generation using Diffusion Models",
    author="Jeffrey Oduro Asante",
    author_email="jeffaoduro@gmail.com",
    packages=find_packages(),
    install_requires=[
        "jax>=0.4.27",
        "flax>=0.10.3",
        "optax>=0.2.4",
        "numpy>=1.23.2",
        "Pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "tensorflow>=2.15.0",  # For dataset handling
        "orbax-checkpoint>=0.11.6",  # For checkpoint management
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
