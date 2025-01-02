import random
import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import scanpy as sc
from tqdm import tqdm
from TFcomb.tools.tf_inference import get_directing_score
# import celloracle as co

# def test3():
#     print(settings['save_figure_as'])

def fix_seed(seed):
    """
    Seed all necessary random number generators.
    """
    if seed is None:
        seed = random.randint(1, 10000)
    torch.set_num_threads(1)  # Suggested for issues with deadlocks, etc.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True 


def add_branches(adata,
                 adata_stream,
                 root=None,
                 branches=None,
                 branch_celltype=None,
                 step=3,
                 filter_branch=True,
                 root_celltype=None):
    '''
    this function is used to add the branches that stream inferred.
    
    Parameters:
    ----------
    adata: anndata. the processed anndata
    adata_stream: anndata. input the adata to stream to get the adata_stream
    root: str. the start branch 
    branches: [str]. the developed branches.
    branch_celltype: [str]. the celltype that each branch corresponds to.
    step: int. the fragments that each branch is split.
    filter_branch: Bool. Whether filter the wrong celltype on the branch.
    root_celltype: str. the celltype name of the root.
    
    Return:
    adata: anndata. the anndata added the branches information.
    ----------
    '''
    
    # get the special columns of adata_stream
    stream_columns = np.setdiff1d(adata_stream.obs.columns,adata.obs.columns)
    # combine the adata_stream obs to adata
    df = pd.merge(left=adata.obs,right=adata_stream.obs[stream_columns],how='left',left_index=True,right_index=True)
    adata.obs = df
    
    # add column branches
    adata.obs['branches'] = [i[2:4]+'-'+i[8:10] for i in  adata.obs['branch_id_alias']]
    
    # this function is used to set the interval number
    def set_interval(length,step):
        interval = int(np.floor(length/step))
        return [interval]*(step-1)+[length-interval*(step-1)]
    
    # get the branch_stage
    index_list, branch_stage_list = [], []
    for branch in branches:
        branch_alias = branch+'-'+root
        adata_sub = adata[adata.obs.branches==branch_alias]

        df = adata_sub.obs[['{}_pseudotime'.format(root)]]
        df = df.sort_values(by='{}_pseudotime'.format(root))
        index_list = index_list+list(df.index)


        interval_list = set_interval(len(df),step)
        for (i,interval) in enumerate(interval_list):
            branch_stage_list = branch_stage_list+[branch_alias+'-'+str(i)]*interval

    # combine the branch stage with adata.obs
    df = pd.DataFrame({'branch_stage':branch_stage_list})
    df.index = index_list

    df = pd.merge(left=adata.obs,right=df,how='left',left_index=True,right_index=True)
    adata.obs = df
    
    # add the branch_celltype. we first test the branch_celltype, which is is simple to compare with the DE analysis
    adata.obs['branch_celltype'] = adata.obs[['branches', 'celltype']].agg('-'.join, axis=1)
    
    # filter the wrong celltype on the branch
    if filter_branch:      
        branch_dict = dict(zip(branches, branch_celltype))
        obs_list = []
        for i,name in enumerate(list(adata.obs.branch_stage)):
            for key,value in branch_dict.items():
                if key in name:
                    if adata.obs.celltype[i] in [root_celltype,value]:
                        obs_list.append(adata.obs_names[i])
        adata = adata[obs_list]
    
    return adata
    
def filter_branch(adata,
                  root=None,
                  branches=None,
                  branch_celltype=None,
                  root_celltype=None):
    branch_dict = dict(zip(branches, branch_celltype))
    obs_list = []
    for i,name in enumerate(list(adata.obs.branch_stage)):
        for key,value in branch_dict.items():
            if key in name:
                if adata.obs.celltype[i] in [root_celltype,value]:
                    obs_list.append(adata.obs_names[i])
    adata = adata[obs_list]
    
    return adata



def adjust_coef(coef,
                tf_index,
                init_ave,
                total_rs_dict):
    '''
    used to adjust the regression model coefficients
    
    
    '''
    scaler = MinMaxScaler((1,2))
    init_ave_scale = scaler.fit_transform(init_ave)
    
    tf_rs = np.log(np.array(list(total_rs_dict.values()))+1)
    scaler = MinMaxScaler((1,2))
    tf_rs = scaler.fit_transform(tf_rs.reshape(-1,1)).ravel()
    
    adj_coef = (coef)/(init_ave_scale.ravel()[tf_index])*(tf_rs.ravel())
    
    return adj_coef

def get_de_genes(adata,
                 cluster_name_for_GRN_unit,
                 init_cluster,
                 control_cluster,
                 tf_list,
                 p_val=5e-2):
    '''
    this function is used to get the pos and neg differential genes in adata.
    
    '''
    
    group = control_cluster
    adata_part = adata[adata.obs[cluster_name_for_GRN_unit].isin([init_cluster, control_cluster])]
    sc.tl.rank_genes_groups(adata=adata_part, groupby = cluster_name_for_GRN_unit,groups=[group], reference='rest', method='wilcoxon')
    # p_val = 5e-2
    idx=np.where(adata_part.uns['rank_genes_groups']['pvals_adj'][group]>p_val)[0][0]
    # idx = len(adata_part.uns['rank_genes_groups']['names'][group])
    pos_gene = adata_part.uns['rank_genes_groups']['names'][group][0:idx]
    pos_gene_total = adata_part.uns['rank_genes_groups']['names'][group]

    idx=np.where(adata_part.uns['rank_genes_groups']['pvals_adj'][::-1][group]>p_val)[0][0]
    # idx = len(adata_part.uns['rank_genes_groups']['names'][group])
    neg_gene = adata_part.uns['rank_genes_groups']['names'][::-1][group][0:idx]
    neg_gene_total = adata_part.uns['rank_genes_groups']['names'][::-1][group]

    pos_gene_tf = [i for i in pos_gene if i in tf_list]
    neg_gene_tf = [i for i in neg_gene if i in tf_list]
    
    return pos_gene, neg_gene, pos_gene_tf, neg_gene_tf


def import_TF_data(TF_info_matrix=None, TF_info_matrix_path=None, TFdict=None):
    """
    Load data about potential-regulatory TFs.
    You can import either TF_info_matrix or TFdict.
    For more information on how to make these files, please see the motif analysis module within the celloracle tutorial.

    Args:
        TF_info_matrix (pandas.DataFrame): TF_info_matrix.

        TF_info_matrix_path (str): File path for TF_info_matrix (pandas.DataFrame).

        TFdict (dictionary): Python dictionary of TF info.
    """

#     if self.adata is None:
#         raise ValueError("Please import scRNA-seq data first.")

#     if len(self.TFdict) != 0:
#         print("TF dict already exists. The old TF dict data was deleted. \n")

    if not TF_info_matrix is None:
        tmp = TF_info_matrix.copy()
        tmp = tmp.drop(["peak_id"], axis=1)
        tmp = tmp.groupby(by="gene_short_name").sum()
        TFdict = dict(tmp.apply(lambda x: x[x>0].index.values, axis=1))

    if not TF_info_matrix_path is None:
        tmp = pd.read_parquet(TF_info_matrix_path)
        tmp = tmp.drop(["peak_id"], axis=1)
        tmp = tmp.groupby(by="gene_short_name").sum()
        TFdict = dict(tmp.apply(lambda x: x[x>0].index.values, axis=1))

#     if not TFdict is None:
#         self.TFdict=TFdict.copy()

#     # Update summary of TFdata
#     self._process_TFdict_metadata()
    return TFdict


def get_benchmark_score(gt_list, tf_list):
    total_score = 0
    tmp = 0
    for i, tf in enumerate(tf_list):
        if tf in gt_list:
            total_score = total_score + 1/len(gt_list)*(1-(i-tmp)/len(tf_list))
            tmp = tmp + 1 # this value is used to calculate how many tfs have been inferenced
    return total_score


def get_percentile_thre(df, value, fillna_with_zero, cluster1, cluster2, TF_number):
    piv = pd.pivot_table(df, values=value, columns="cluster", index="index")
    if fillna_with_zero:
        piv = piv.fillna(0)
    else:
        piv = piv.fillna(piv.mean(axis=0))
    percentile_up = 100
    percentile_down = 50
    percentile = (percentile_up+percentile_down)/2
    TF_number_tmp = 0
    count = 0
    while(TF_number_tmp!=TF_number):
    # for i in range(20):
        count = count+1
        goi1 = piv[piv[cluster1] > np.percentile(piv[cluster1][piv[cluster1].values>0].values, percentile)].index
        goi2 = piv[piv[cluster1] < np.percentile(piv[cluster1][piv[cluster1].values<0].values, 100-percentile)].index
        goi3 = piv[piv[cluster2] > np.percentile(piv[cluster2].values, percentile)].index
        goi4 = piv[piv[cluster1] > 0].index
        TF_number_tmp = len(np.intersect1d(goi4,np.union1d(goi1, goi3)))
        # print(TF_number_tmp,percentile)

        if TF_number_tmp>TF_number:
            percentile_down = percentile
            percentile = (percentile_up+percentile)/2

        elif TF_number_tmp<TF_number:
            percentile_up = percentile
            percentile = (percentile_down+percentile)/2
            
        if count>20 and TF_number_tmp!=TF_number:
            print(f'percentile search failed and the TF number is {TF_number_tmp}')
            break
    print(f'the percentile threshold is {percentile}')
    
    return percentile


def set_plot_para():
    
    import matplotlib as mpl
    mpl.rcParams["font.sans-serif"]=["Arial"]
    mpl.rcParams["axes.unicode_minus"]=False
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['text.color'] = 'black'

    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['axes.titlecolor'] = 'black'
    plt.rcParams['legend.labelcolor'] = 'black'
    
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


def group_hvg(adata_anno, cluster_name_for_GRN_unit, init_cluster, control_cluster, total_tf_list):
    adata = adata_anno.copy()
    adata.var_names_make_unique()
    gene_num = int(adata.shape[1]*0.95)
    group = control_cluster
    adata = adata[adata.obs[cluster_name_for_GRN_unit].isin([init_cluster, control_cluster])].copy()

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # Normalize gene expression matrix with total UMI count per cell
    sc.pp.normalize_per_cell(adata, key_n_counts='n_counts_all')
    adata_raw = adata.copy()

    sc.pp.log1p(adata)

    df = sc.pp.highly_variable_genes(adata,n_top_genes=gene_num,inplace=False)
    df = df.sort_values(by='dispersions_norm',ascending=False)
    idx_list = list(df[df.highly_variable==True].index)

    gene_list = list(adata[:, idx_list].var_names)

    adata_tf_list2 = [i for i in gene_list if i in total_tf_list]
    print(f'group_hvg: tf in adata is: {len(adata_tf_list2)}')

    return adata_tf_list2

def group_marker_gene(adata_anno, cluster_name_for_GRN_unit, init_cluster, control_cluster, total_tf_list):
    
    adata = adata_anno.copy()
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    group = control_cluster
    adata_part = adata[adata.obs[cluster_name_for_GRN_unit].isin([init_cluster, control_cluster])]
    sc.tl.rank_genes_groups(adata=adata_part, groupby = cluster_name_for_GRN_unit,groups=[group], reference='rest', method='wilcoxon')
    # p_val = 5e-2
    # idx=np.where(adata_part.uns['rank_genes_groups']['pvals_adj'][group]>p_val)[0][0]
    idx = len(adata_part.uns['rank_genes_groups']['names'][group])
    gene_list = adata_part.uns['rank_genes_groups']['names'][group][0:idx]

    adata_tf_list3 = [i for i in gene_list if i in total_tf_list]
    print(f'group_marker_gene: tf in adata is: {len(adata_tf_list3)}')
    
    return adata_tf_list3

def group_highexp_gene(adata_anno, cluster_name_for_GRN_unit, init_cluster, control_cluster, total_tf_list, subset = False):
    
    adata = adata_anno.copy()
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    if subset:
        adata = adata[adata.obs[cluster_name_for_GRN_unit].isin([control_cluster])].copy()
    
    df = pd.DataFrame({'gene':adata.var_names,
                       'score':np.array(np.mean(adata.X,axis=0)).ravel()})
    df = df.sort_values(by='score', ascending=False)
    gene_list = list(df.iloc[0:len(df),:].gene)

    adata_tf_list4 = [i for i in gene_list if i in total_tf_list]
    print(f'group_highexp_gene: tf in adata is: {len(adata_tf_list4)}')

    return adata_tf_list4


def select_add_TF(adata_anno, cluster_name_for_GRN_unit, init_cluster, control_cluster, base_GRN_dir, subset = False):
    # result_dir = "/nfs/public/lichen/results/ISDE_GRN/SHARE-seq/"
    # prefix = 'skin'
    base_GRN = pd.read_parquet(base_GRN_dir)
    TFdict = import_TF_data(TF_info_matrix=base_GRN)
    tf_target_dict = {}
    for target, gene_set in TFdict.items():
        for tf in gene_set:
            if tf not in tf_target_dict:
                tf_target_dict[tf] = []
                tf_target_dict[tf].append(target)
            else:
                tf_target_dict[tf].append(target)
    total_tf_list = list(tf_target_dict.keys())
    
    adata_tf_list2 = group_hvg(adata_anno, cluster_name_for_GRN_unit, init_cluster, control_cluster, total_tf_list)
    adata_tf_list3 = group_marker_gene(adata_anno, cluster_name_for_GRN_unit, init_cluster, control_cluster, total_tf_list)
    adata_tf_list4 = group_highexp_gene(adata_anno, cluster_name_for_GRN_unit, init_cluster, control_cluster, total_tf_list, subset)
    
    add_TF_list = np.intersect1d(np.intersect1d(adata_tf_list3[0:100],adata_tf_list4[0:100]),adata_tf_list2[0:400])
    print(f'add_TF_list length is: {len(add_TF_list)}')
    # tmp_list = np.intersect1d(adata_tf_list3[0:300],adata_tf_list4[0:300])
    # len(tmp_list), [i in tmp_list for i in filter_tforf_list],sorted(tmp_list)
    
    return add_TF_list


def get_single_TF(
    tf_list,
    rr_corr,
    TF_ds_dict,
    direction = 'pos'):
    if direction == 'pos':
        tf_direction_list = [i for i in tf_list if rr_corr.coef_[tf_list.index(i)] > 0]
    elif direction == 'neg':
        tf_direction_list = [i for i in tf_list if rr_corr.coef_[tf_list.index(i)] < 0]
    
    df_score = pd.DataFrame({
        'TF': tf_direction_list,
        'directing_score': [TF_ds_dict[tf] for tf in tf_direction_list],
        'expected_alteration': [rr_corr.coef_[tf_list.index(tf)] for tf in tf_direction_list]
    })
    df_score = df_score.sort_values(by='directing_score', ascending=False).reset_index(drop=True)
    return df_score


def get_multi_TF(
    tf_list,
    rr_corr,
    tf_GRN_mtx,
    tf_GRN_dict,
    source_ave,
    target_ave,
    direction,
    number):
    
    from itertools import combinations
    
    if direction == 'pos':
        tf_direction_list = [i for i in tf_list if rr_corr.coef_[tf_list.index(i)] > 0]
    elif direction == 'neg':
        tf_direction_list = [i for i in tf_list if rr_corr.coef_[tf_list.index(i)] < 0]
    

    TF_combination_dict = {}
    for TF_combination in tqdm(list(combinations(tf_direction_list,number))):
        TF_pcc = get_directing_score(change_tf = list(TF_combination),
                                      rr = rr_corr,
                                      tf_GRN_mtx = tf_GRN_mtx,
                                      diff_ave = (target_ave - source_ave).ravel(),
                                      mode='multi',
                                      if_print=False,
                                      tf_GRN_dict = tf_GRN_dict,
                                      X=None)
        TF_combination_dict[TF_combination] = TF_pcc
    df_score = pd.DataFrame({'TF_combination':TF_combination_dict.keys(),
                       'directing_score':TF_combination_dict.values()})
    df_score = df_score.sort_values(by='directing_score', ascending=False).reset_index(drop=True)
    return df_score