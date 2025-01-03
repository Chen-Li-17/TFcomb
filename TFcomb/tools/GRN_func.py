import pandas as pd
import numpy as np

# def get_GRN_parameters(oracle,
#                        links,
#                        if_print=True):
#     '''
#     this function is used to get the dicts calculated from the GRN, which is required in the downstream analysis
    
#     Parameters:
#     ----------
#     oracle: oracle object.
#     links: celloracle links object.
    
#     Return:
#     ----------
#     total_gene_GRN_mtx: dict. each value is a df, rows and columns are both total genes.
#     total_tf_GRN_mtx: dict. each value is a df, rows are TFs, columns are total genes.
#     total_tf_GRN_filter_mtx: dict. each value is a df, rows are TFs, columns are regulated genes.
#     total_tf_GRN_dict: dict. each value is a dict, keys are TFs, values are dicts, keys are regulated genes, values are the regulated values.
#     total_tf_info_df: dict. each value is a df, the network scores of TFs, including the total rs and mean rs.
#     total_total_rs_dict: dict. each value is a dict, keys are TFs, values are total regulatory scores.
#     total_mean_rs_dict: dict. each value is a dict, keys are TFs, values are mean regulatory scores.
    
#     '''
#     links.get_network_score()
#     whole_network_score = links.merged_score

#     total_gene_GRN_mtx, total_tf_GRN_mtx, total_tf_GRN_filter_mtx, total_tf_GRN_dict, total_tf_info_df = {},{},{},{},{}
#     total_total_rs_dict, total_mean_rs_dict = {},{}
#     for cluster in links.links_dict.keys():

#         network_score = whole_network_score[whole_network_score.cluster==cluster]
#         #get the gene GRN matrix, and filter the rows that all 0 to get tf GRN mtx
#         gene_GRN_mtx = oracle.coef_matrix_per_cluster[cluster].copy()
#         tf_GRN_mtx = gene_GRN_mtx[~(gene_GRN_mtx == 0).all(axis=1)]

#         total_gene_GRN_mtx[cluster] = gene_GRN_mtx
#         total_tf_GRN_mtx[cluster] = tf_GRN_mtx

#         # get TF-target pair and the regulatory values
#         cluster_tf_dict = {} # the tf to targets
#         for i in range(len(tf_GRN_mtx)):
#             tmp = tf_GRN_mtx.iloc[i,:]
#             tmp = tmp[tmp!=0]

#             cluster_tf_dict[tf_GRN_mtx.index[i]] = {}
#             for j in range(len(tmp)):
#                 cluster_tf_dict[tf_GRN_mtx.index[i]][tmp.index[j]] = tmp.values[j]

#         total_tf_GRN_dict[cluster] = cluster_tf_dict

#         #cal total/averate regulatory score (abs)
#         cluster_total_rs_dict, cluster_mean_rs_dict = {}, {} # regulatory score
#         for key, value in cluster_tf_dict.items():
#             cluster_total_rs_dict[key] = np.sum(np.abs(list(value.values())))
#             cluster_mean_rs_dict[key] = np.average(np.abs(list(value.values())))
#         total_total_rs_dict[cluster] = cluster_total_rs_dict
#         total_mean_rs_dict[cluster] = cluster_mean_rs_dict

#         #--------- generate regulatory score df
#         cluster_rs_df = pd.DataFrame({'gene':list(cluster_total_rs_dict.keys()),
#                                      'total_rs':list(cluster_total_rs_dict.values()),
#                                      'mean_rs':list(cluster_mean_rs_dict.values())})
#         cluster_rs_df.index = cluster_rs_df.gene

#         cluster_tf_info_df = pd.merge(left=cluster_rs_df,right=network_score,how='inner',left_index=True,right_index=True)
#         total_tf_info_df[cluster] = cluster_tf_info_df

#         tf_GRN_filter_mtx = tf_GRN_mtx.loc[:,tf_GRN_mtx.any()]
#         total_tf_GRN_filter_mtx = tf_GRN_filter_mtx
#         if if_print:
#             print('==============',cluster)
#             print("tf_GRN_filter_mtx shape is:",tf_GRN_filter_mtx.shape)
            
#     return total_gene_GRN_mtx, total_tf_GRN_mtx, total_tf_GRN_filter_mtx, total_tf_GRN_dict, total_tf_info_df, total_total_rs_dict, total_mean_rs_dict
    
def get_GRN_parameters(oracle,
                       combine,
                       ):
    """
    Extracts GRN (Gene Regulatory Network) parameters for downstream analysis.

    This function processes the Oracle object's GRN data and calculates key GRN-related 
    matrices and dictionaries required for further analysis.

    Args:
        oracle (co.Oracle): The Oracle object containing the GRN data.
        combine (str): The specific cluster or condition to extract GRN information from.

    Returns:
        tuple: A tuple containing the following:
            - gene_GRN_mtx_ori (pd.DataFrame): Original GRN matrix with all genes as rows and columns.
            - tf_GRN_mtx_ori (pd.DataFrame): Filtered GRN matrix with TFs (transcription factors) as rows and all genes as columns.
            - tf_GRN_dict (dict): Dictionary where keys are TFs and values are dictionaries. 
              The nested dictionary maps target genes to their regulatory scores.
    """
    gene_GRN_mtx_ori = oracle.coef_matrix_per_cluster[combine].copy()
    tf_GRN_mtx_ori = gene_GRN_mtx_ori[~(gene_GRN_mtx_ori == 0).all(axis=1)]

    # - get TF-target pair and the regulatory values
    tf_GRN_dict = {} # the tf to targets
    for i in range(len(tf_GRN_mtx_ori)):
        tmp = tf_GRN_mtx_ori.iloc[i,:]
        tmp = tmp[tmp!=0]
        tf_GRN_dict[tf_GRN_mtx_ori.index[i]] = {}
        for j in range(len(tmp)):
            tf_GRN_dict[tf_GRN_mtx_ori.index[i]][tmp.index[j]] = tmp.values[j]
            
    return gene_GRN_mtx_ori, tf_GRN_mtx_ori, tf_GRN_dict
    