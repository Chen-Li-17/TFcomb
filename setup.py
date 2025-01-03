from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name = "TFcomb",
    version = "1.0",
    keywords = ["pip", "tfcomb"],
    description = "TFcomb is a python library to identify reprogramming TFs and TF combinations using scRNA-seq and scATAC-seq data.",
    long_description = "TFcomb is a python library to identify reprogramming TFs and TF combinations using scRNA-seq and scATAC-seq data.",
    license = "MIT License",
    url = "https://github.com/Chen-Li-17/TFcomb",
    author = "Chen Li",
    author_email = "chen-li21@qq.com",
    packages = ['TFcomb'],
    python_requires = ">3.8.0",
    classifiers = [
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
    install_requires=required
)
