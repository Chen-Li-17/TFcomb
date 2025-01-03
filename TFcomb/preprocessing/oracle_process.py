import matplotlib.pyplot as plt
import numpy as np


def oracle_preprocess(oracle,
                      k=10):
    """
    Preprocesses the Oracle object by performing PCA and KNN imputation.

    This function computes the optimal number of principal components based on 
    the explained variance ratio, adjusts the number of neighbors (k) for KNN 
    imputation, and applies the preprocessing steps to the Oracle object.

    Args:
        oracle (co.Oracle): The Oracle object containing the data to preprocess.
        k (int, optional): The number of neighbors for KNN imputation. Defaults to 10.

    Returns:
        co.Oracle: The Oracle object after PCA and KNN imputation.
    """
    oracle.perform_PCA()
    plt.plot(np.cumsum(oracle.pca.explained_variance_ratio_)[:100])
    n_comps = np.where(np.diff(np.diff(np.cumsum(oracle.pca.explained_variance_ratio_))>0.002))[0][0]
    plt.axvline(n_comps, c="k")
    plt.show()
    print('n_comps is:',n_comps)
    n_comps = min(n_comps, 50)
    n_cell = oracle.adata.shape[0]
    print(f"cell number is :{n_cell}")
    k_auto = int(0.025*n_cell)
    print(f"Auto-selected k is :{k_auto}")
    print("but we set default k is:",k)
    oracle.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True, b_sight=k*8,
                      b_maxl=k*4, n_jobs=4)
    return oracle