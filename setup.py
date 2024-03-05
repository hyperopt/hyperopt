import re

import setuptools

with open("hyperopt/__init__.py", encoding="utf8") as f:
    version = re.search(r"__version__ = \"(.*?)\"", f.read()).group(1)
    if version is None:
        raise ImportError("Could not find __version__ in hyperopt/__init__.py")

setuptools.setup(
    name="hyperopt",
    version=version,
    packages=setuptools.find_packages(include=["hyperopt*"]),
    entry_points={"console_scripts": ["hyperopt-mongo-worker=hyperopt.mongoexp:main"]},
    url="https://hyperopt.github.io/hyperopt",
    project_urls={
        "Source": "https://github.com/hyperopt/hyperopt",
    },
    author="James Bergstra",
    author_email="james.bergstra@gmail.com",
    description="Distributed Asynchronous Hyperparameter Optimization",
    long_description="",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Environment :: Console",
        "License :: OSI Approved :: BSD License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    platforms=["Linux", "OS-X", "Windows"],
    license="BSD",
    keywords="Bayesian optimization hyperparameter model selection",
    include_package_data=True,
    requires_python=">=3.7",
    install_requires=[
        "cloudpickle",
        "networkx>=2.2",
        "numpy>=1.17",
        "scipy",
        "tqdm",
    ],
    extras_require={
        "SparkTrials": ["pyspark", "py4j"],
        "MongoTrials": "pymongo>=4.0.0",
        "ATPE": ["lightgbm", "scikit-learn"],
        "dev": ["black", "pre-commit", "pytest"],
    },
    tests_require=["pytest"],
    zip_safe=False,
)
