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
    """
    Identifies differentially expressed genes (DEGs) between two clusters in an `AnnData` object.

    This function performs differential expression analysis between the `init_cluster` and the `control_cluster`, 
    returning genes that are upregulated (positive) and downregulated (negative) in the `init_cluster` relative to the 
    `control_cluster`, as well as those that are transcription factors (TFs).

    Args:
        adata (anndata.AnnData): Annotated data matrix containing the gene expression data.
        cluster_name_for_GRN_unit (str): The column name in `adata.obs` that contains cluster labels.
        init_cluster (str): The name of the initial cluster to compare.
        control_cluster (str): The name of the control cluster to compare.
        tf_list (list): List of known transcription factors (TFs) to filter the DEGs.
        p_val (float, optional): The adjusted p-value threshold for determining significant genes. Defaults to 5e-2.

    Returns:
        tuple: A tuple containing four lists:
            - `pos_gene` (list): Genes upregulated in `init_cluster` relative to `control_cluster`.
            - `neg_gene` (list): Genes downregulated in `init_cluster` relative to `control_cluster`.
            - `pos_gene_tf` (list): Transcription factors among the upregulated genes.
            - `neg_gene_tf` (list): Transcription factors among the downregulated genes.
    """
    
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
    """
    Computes the benchmark score based on the ranking of predicted transcription factors (TFs) 
    compared to the ground truth list of TFs.

    The score rewards TFs that appear earlier in the predicted list if they are present in the ground truth list.
    The benchmark score is based on a weighted sum where the weight decreases as the position in the 
    predicted list increases.

    Args:
        gt_list (list): A list of ground truth transcription factors (TFs), which represents the correct order of TFs.
        tf_list (list): A list of predicted transcription factors (TFs) based on the model.

    Returns:
        float: The benchmark score indicating the quality of the prediction. The score is between 0 and 1, 
               with higher values indicating better performance.
    """
    total_score = 0
    tmp = 0
    for i, tf in enumerate(tf_list):
        if tf in gt_list:
            total_score = total_score + 1/len(gt_list)*(1-(i-tmp)/len(tf_list))
            tmp = tmp + 1 # this value is used to calculate how many tfs have been inferenced
    return total_score


def get_percentile_thre(df, value, fillna_with_zero, cluster1, cluster2, TF_number):
    """
    Determines the optimal percentile threshold to select a specific number of transcription factors (TFs) 
    based on their expression levels in two clusters.

    This function calculates the percentile threshold that selects a set of TFs whose number is closest to the 
    desired `TF_number`, based on their expression in two clusters (`cluster1` and `cluster2`). It iteratively 
    adjusts the threshold until the number of selected TFs matches `TF_number`.

    Args:
        df (pandas.DataFrame): A dataframe containing the expression data for genes. It should have at least 
            the following columns: `index`, `cluster`, and the column specified by `value` representing 
            expression values.
        value (str): The name of the column in `df` that contains the expression values for the genes.
        fillna_with_zero (bool): If `True`, fills NaN values with zeros. If `False`, fills NaN values with 
            the mean expression across clusters.
        cluster1 (str): The name of the first cluster to use for selecting TFs.
        cluster2 (str): The name of the second cluster to use for selecting TFs.
        TF_number (int): The target number of TFs to select based on their expression.

    Returns:
        float: The percentile threshold that results in selecting `TF_number` TFs.
    """
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


def select_add_TF(adata_anno, cluster_name_for_GRN_unit, init_cluster, control_cluster, base_GRN_dir, subset = False,
                  filter_num_1 = 100,
                  filter_num_2 = 100,
                  filter_num_3 = 400):
    """
    Selects additional transcription factors (TFs) based on different gene grouping criteria from the given data.

    The function performs multiple filtering steps based on various conditions such as highly variable genes,
    marker genes, and high-expression genes. It then returns a list of TFs that satisfy the filtering criteria 
    across these groupings.

    Args:
        adata_anno (AnnData): Annotated data matrix containing gene expression and metadata.
        cluster_name_for_GRN_unit (str): The name of the cluster column in the `adata_anno` object.
        init_cluster (str): The name of the initial cluster to be compared in the analysis.
        control_cluster (str): The name of the control cluster to be compared in the analysis.
        base_GRN_dir (str): The directory path for the base GRN file in Parquet format.
        subset (bool, optional): Whether to use a subset of the data for filtering. Default is False.
        filter_num_1 (int, optional): Number of TFs to be filtered based on marker gene grouping. Default is 100.
        filter_num_2 (int, optional): Number of TFs to be filtered based on high-expression gene grouping. Default is 100.
        filter_num_3 (int, optional): Number of TFs to be filtered based on highly variable gene grouping. Default is 400.

    Returns:
        list: A list of transcription factors (TFs) that satisfy all the filtering conditions.
    """
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
    
    add_TF_list = np.intersect1d(np.intersect1d(adata_tf_list3[0:filter_num_1],adata_tf_list4[0:filter_num_2]),adata_tf_list2[0:filter_num_3])
    print(f'add_TF_list length is: {len(add_TF_list)}')
    # tmp_list = np.intersect1d(adata_tf_list3[0:300],adata_tf_list4[0:300])
    # len(tmp_list), [i in tmp_list for i in filter_tforf_list],sorted(tmp_list)
    
    return add_TF_list


def get_single_TF(
    tf_list,
    rr_corr,
    TF_ds_dict,
    direction = 'pos'):
    """
    Retrieves a list of transcription factors (TFs) that are either positively or negatively correlated 
    with the expected alteration and ranks them based on their directing score.

    This function filters the list of TFs based on their coefficients in the regression model (`rr_corr`) 
    and returns a DataFrame containing the selected TFs along with their directing scores and expected alterations.

    Args:
        tf_list (list): A list of transcription factors (TFs) to be considered for analysis.
        rr_corr (sklearn.linear_model.Ridge): A fitted regression model with coefficients for each TF in `tf_list`.
        TF_ds_dict (dict): A dictionary where keys are TFs and values are their directing scores.
        direction (str, optional): The direction of TF correlation to filter by. 
            - 'pos' (default): Selects TFs with positive correlation (coefficients > 0).
            - 'neg': Selects TFs with negative correlation (coefficients < 0).

    Returns:
        pandas.DataFrame: A DataFrame containing the following columns:
            - 'TF': The transcription factor.
            - 'directing_score': The directing score for each TF from `TF_ds_dict`.
            - 'expected_alteration': The expected alteration value (i.e., the regression coefficient for each TF).
    """
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
    """
    Retrieves and ranks all possible combinations of transcription factors (TFs) of a specified size 
    (number of TFs) based on their directing scores.

    This function considers TFs based on the direction of their coefficients in the regression model 
    (`rr_corr`) and computes their directing score for all combinations of a specified size.

    Args:
        tf_list (list): A list of transcription factors (TFs) to be considered for analysis.
        rr_corr (sklearn.linear_model.Ridge): A fitted regression model with coefficients for each TF in `tf_list`.
        tf_GRN_mtx (pandas.DataFrame): A matrix of TF-gene interactions used to calculate directing scores.
        tf_GRN_dict (dict): A dictionary containing the TF-gene relationships used for score calculations.
        source_ave (numpy.ndarray): The average expression of the source cluster (initial state).
        target_ave (numpy.ndarray): The average expression of the target cluster (control state).
        direction (str): The direction of TF correlation to filter by.
            - 'pos': Selects TFs with positive coefficients.
            - 'neg': Selects TFs with negative coefficients.
        number (int): The number of TFs to combine in each group.

    Returns:
        pandas.DataFrame: A DataFrame containing the following columns:
            - 'TF_combination': A tuple of TFs in the combination.
            - 'directing_score': The directing score for each TF combination, which reflects the predicted effect 
              of the combination on gene expression.
    """
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