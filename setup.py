import setuptools
from m3m import __VERSION__

print("VERSION: ".format(__VERSION__))

with open("README.md", "r") as fh:
    long_description = fh.read()

packages = setuptools.find_packages()
print(packages)

setuptools.setup(
    name="m3m",
    version=__VERSION__,
    author="Weijie Liu",
    author_email="autoliuweijie@163.com",
    description="The pipy version of FastPLM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/autoliuweijie/m3m",
    packages=packages,
    # package_data=package_data,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.4',
    install_requires=[
        'torch>=1.0.0',
        ]
)
