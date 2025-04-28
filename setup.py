from setuptools import setup, find_packages


with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="reinforceui-studio",
    version="1.3.1",
    author="David Valencia",
    description="A GUI to simplify the configuration and monitoring of RL training processes.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dvalenciar/ReinforceUI-Studio",
    license="MIT License",
    author_email="support@reinforceui-studio",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "reinforceui-studio=main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    include_package_data=True,
)
