.. _install:

Installation
=============

We tested TFcomb with the Python environment 3.9.
::
    conda create --name TFcomb python=3.9


Prerequisites
-------------
Before installing TFcomb, users should first install Pytorch and DGL.

Check your CUDA version and install Pytorch with the right version. For example,
the CUDA version of our server is 11.3, so we refered the `website <https://pytorch.org/get-started/previous-versions/>`_ and 
install Pytorch with:
::
    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

The versions of DGL can be refered at the `website <https://anaconda.org/dglteam/dgl>`_.
According to our CUDA version, we installed DGL with:
::
    conda install dglteam/label/cu113::dgl

By the way, if the install speed of conda is too slow, you can just install mamba and replace conda with
mamba:
::
    conda install mamba
    mamba install dglteam/label/cu113::dgl

After Pytorch and DGL are installed in the environment, you can install
TFcomb through pip or github.


PyPI
----

TFcomb is available on PyPI here_ and can be installed via::

    pip install TFcomb


GitHub
--------

PAST can also installed from GitHub via::

    git clone https://github.com/Chen-Li17/xxx.git
    cd TFcomb
    python setup.py install

Dependency of test env
-----------
::

    adjustText==0.8
    auto_mix_prep==0.2.0
    celloracle==0.12.0
    dgl==1.1.1
    gseapy==1.1.3
    matplotlib==3.6.0
    networkx==3.1
    numpy==1.22.4
    pandas==1.5.3
    scanpy==1.10.3
    scikit_learn==1.2.2
    scipy==1.5.3
    seaborn==0.13.2
    torch==1.12.1
    tqdm==4.65.0
    umap_learn==0.5.3
    ipykernel==6.29.5

If there are any problems with your installation, you can refer to the specific versions of the packages.

.. _here: https://pypi.org/project/xxx


