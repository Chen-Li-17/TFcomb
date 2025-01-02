from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name = "TFcomb",
    version = "0.1",
    keywords = ["pip", "tfcomb"],
    description = "TFcomb xxx",
    long_description = "xxx",
    license = "MIT License",
    url = "https://github.com/Chen-Li-17/TFcomb",
    author = "Chen Li",
    author_email = "chen-li21@qq.com",
    packages = find_packages(),
    python_requires = ">3.8.0",
    classifiers = [
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
    install_requires=required
)
