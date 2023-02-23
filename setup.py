import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="eletor",
    version="0.0.1",
    author="Javier Clavijo over Machiel Bos's work",
    author_email="jclavijo@fi.uba.ar",
    description="A collection of programs simulate geodetic time series with some modifications form J Clavijo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jjclavijo/apio-divertido.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'bson',
        'hectorp'
    ]
)
