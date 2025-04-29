from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="reinforceui-studio",
    version="1.3.1",
    author="David Valencia",
    author_email="support@reinforceui-studio.com",
    description="A GUI to simplify the configuration and monitoring of RL training processes.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dvalenciar/ReinforceUI-Studio",
    license="MIT",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "reinforceui-studio=main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: X11 Applications :: Qt",
    ],
    keywords="reinforcement-learning machine-learning deep-learning GUI",
    project_urls={
        "Documentation": "https://docs.reinforceui-studio.com",
        "Repository": "https://github.com/dvalenciar/ReinforceUI-Studio",
        "Tracker": "https://github.com/dvalenciar/ReinforceUI-Studio/issues",
    },
    python_requires=">=3.10",
    include_package_data=True,
)
