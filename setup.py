import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="geobind",
    version="1.0",
    author="Jared Sagendorf",
    author_email="jsagendorf@gmail.com",
    description="A package for predicting protein binding sites and binding function using Geometric Deep Learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jaredsagendorf/geobind",
    packages=setuptools.find_packages(),
    classifiers=[
	"Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
