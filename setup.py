from setuptools import setup, find_packages

setup(
    name="polyscribe",
    version="0.1.0",
    description="Polyphonic Audio-to-MIDI CLI Converter for Termux, Linux, and macOS",
    author="VibeCoder",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "polyscribe = polyscribe.cli:main",
        ],
    },
    install_requires=[
        "numpy>=1.20.0",
        "mido>=1.2.10",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    ],
)
