import pandas as pd
import numpy as np

def _adata_to_matrix(adata, layer_name, transpose=True):
    """
    Extract an numpy array from adata and returns as numpy matrix.

    Args:
        adata (anndata): anndata

        layer_name (str): name of layer in anndata

        trabspose (bool) : if True, it returns transposed array.

    Returns:
        2d numpy array: numpy array
    """
    if isinstance(adata.layers[layer_name], np.ndarray):
        matrix = adata.layers[layer_name].copy()
    else:
        matrix = adata.layers[layer_name].todense().A.copy()

    if transpose:
        matrix = matrix.transpose()

    return matrix.copy(order="C")


def _adata_to_df(adata, layer_name, transpose=False):
    """
    Extract an numpy array from adata and returns as pandas DataFrane with cell names and gene names.

    Args:
        adata (anndata): anndata

        layer_name (str): name of layer in anndata

        trabspose (bool) : if True, it returns transposed array.

    Returns:
        pandas.DataFrame: data frame (cells x genes (if transpose == False))
    """
    array = _adata_to_matrix(adata, layer_name, transpose=False)
    df = pd.DataFrame(array, columns=adata.var.index.values, index=adata.obs.index.values)

    if transpose:
        df = df.transpose()
    return df