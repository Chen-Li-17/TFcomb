import celloracle as co
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import scipy
# from trajectory.oracle_utility import _adata_to_df
from TFcomb.tools.utils_celloracle import _adata_to_df
import seaborn as sns

from TFcomb.tools.utils import fix_seed

# def test():
#     fix_seed(0)

def pca_umap_train(adata,
                   cluster_column_name=None,
                   embedding_name=None,
                   n_components=50,
                   svd_solver='arpack',
                   random_seed=2022):
    '''
    Train a pca and umap on the oracle normalized data
    
    Parameters:
    ----------
    adata: anndata. input to the oracle
    cluster_column_name: str. oracle cluster name
    embedding_name: str. oracle embedding name 
    n_components,svd_solver,random_seed: model parameters
    
    Return:
    ----------
    pca_train: trained pca model
    umap_train: trained umap model
    '''
    
    plt.rcParams["figure.figsize"] = [5, 5]
    oracle = co.Oracle()
    adata.X = adata.layers["raw_count"].copy()
    oracle.import_anndata_as_raw_count(adata=adata,
                                       cluster_column_name=cluster_column_name,
                                       embedding_name=embedding_name)
    
    gem_imputed = _adata_to_df(oracle.adata, "normalized_count")
    pca_train=PCA(n_components=n_components, svd_solver=svd_solver,random_state=random_seed)
    X_pca = pca_train.fit_transform(gem_imputed.values)
    umap_train=umap.UMAP(random_state=random_seed)
    proj = umap_train.fit_transform(X_pca)
    
    return pca_train,umap_train,oracle


def pca_umap_vis(pca_train=None,
                 umap_train=None,
                 exp_mtx=None,
                 label=None,
                 title=None,
                 bbox=1,
                 figsize=(12,5),
                 save=None
                 ):
    if isinstance(exp_mtx,scipy.sparse.csr.csr_matrix):
        exp_mtx = exp_mtx.toarray()
    X_pca = pca_train.transform(exp_mtx)
    proj_ori = umap_train.transform(X_pca)
    fig=plt.figure(figsize=figsize)
    for i in range(1):
        ax_ = fig.add_subplot(1,1,i+1)
        df = {'UMAP_1':proj_ori[:, 0],\
              'UMAP_2':proj_ori[:, 1], \
              'label':label}
        df = pd.DataFrame(df)
        ax = sns.scatterplot(x="UMAP_1", 
                             y="UMAP_2", 
                             hue="label",
                             edgecolor='none',
                             # hue_order=celltypes,
                             # saturation=1,
                             palette = 'tab10', 
                             s=8,linewidth = 0.0001, data=df)
        plt.xticks(rotation=0,fontsize=15)
        plt.yticks(rotation=0,fontsize=15)

        # ax.set(title='original UMAP',xlabel='UMAP_1')
        ax.set_xlabel('')
        ax.set_ylabel('')
        if i>0:
            ax.set_xlim(lim1_x,lim2_x)
            ax.set_ylim(lim1_y,lim2_y)
        ax.set_title(title,fontsize=18)
        axLine, axLabel = ax.get_legend_handles_labels()
        # ax.legend([],[],frameon=False)
        ax.legend(loc='upper right',bbox_to_anchor=(bbox, 1),
                 frameon=False)
    if save:
        fig.savefig(save,facecolor='white',bbox_inches='tight',dpi=400)