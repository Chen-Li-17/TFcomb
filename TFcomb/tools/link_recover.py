from sklearn.model_selection import KFold
import pickle
import os
import numpy as np
from .link_recover_module import *
from .GRN_func import *
from copy import deepcopy

def GAT_recover_links(
                    g = None,
                    save_dir_GNN = None,
                    tf_list = None,
                    gene_list = None,
                    tf_GRN_dict = None,
                    oracle_part = None,
                    links_part = None,
                    combine = None,
                    threshold_number = 10000,
                    alpha_fit_GRN = 10,
                    n_splits = 10,
                    seed = 42,
                    neg_link_split = 'all',
                    model_name = 'GAT',
                    pred_name = 'mlp',
                    hidden_dim = 16,
                    out_dim = 7,
                    num_heads = [4, 4, 6],
                    epochs = 1500,
                    lr = 0.01,
                    patience = 300,
                    device = 'cuda',
                    verbose = True,
                    
                    count_fold = 9,
                    link_score_quantile = 0.1
                    ):
    
    
    model_dir = f'{save_dir_GNN}/model_checkpoint'+'/model.pt'
    pred_dir = f'{save_dir_GNN}/pred_checkpoint'+'/pred.pt'
    os.makedirs(f'{save_dir_GNN}/model_checkpoint', exist_ok=True)
    os.makedirs(f'{save_dir_GNN}/pred_checkpoint', exist_ok=True)
    
    if os.path.exists(os.path.join(save_dir_GNN,'tf_gene_link_dict_Allfold.pickle')):
        with open(os.path.join(f'{save_dir_GNN}/tf_gene_link_dict_Allfold.pickle'), 'rb') as file:
            tf_gene_link_dict_Allfold = pickle.load(file)
        with open(os.path.join(f'{save_dir_GNN}/tf_gene_link_score_dict_Allfold.pickle'), 'rb') as file:
            tf_gene_link_score_dict_Allfold = pickle.load(file)
        # with open(os.path.join(f'{save_dir_GNN}/model_dict_Allfold.pickle'), 'rb') as file:
        #     model_dict_Allfold = pickle.load(file)
        with open(os.path.join(f'{save_dir_GNN}/h_dict_Allfold.pickle'), 'rb') as file:
            h_dict_Allfold = pickle.load(file)
        with open(os.path.join(f'{save_dir_GNN}/thre_dict_Allfold.pickle'), 'rb') as file:
            thre_dict_Allfold = pickle.load(file)
        
    else:

        # train the model
        tf_gene_link_dict_Allfold,tf_gene_link_score_dict_Allfold,\
        model_dict_Allfold,h_dict_Allfold,thre_dict_Allfold\
        =   recover_links(g,
                            gene_list = gene_list,
                            n_splits = n_splits,
                            seed = seed,
                            neg_link_split = neg_link_split,
                            model_name = model_name,
                            pred_name = pred_name,
                            hidden_dim = hidden_dim,
                            out_dim = out_dim,
                            num_heads = num_heads,
                            epochs = epochs,
                            lr = lr,
                            patience = patience,
                            device = device,
                            model_dir = model_dir,
                            pred_dir = pred_dir,
                            verbose = verbose,
                            save_dir = save_dir_GNN,
                            tf_list = tf_list)
    
    # parameters for filter the links
    delete_percent = 0

    # filter the links
    df_tmp, tf_target_link, tf_target_iter_link, tf_recover_link\
        =   filter_recover_links(tf_GRN_dict = tf_GRN_dict,
                                    gene_list = gene_list,
                                    tf_list = tf_list,
                                    tf_gene_link_dict_Allfold = tf_gene_link_dict_Allfold,
                                    tf_gene_link_score_dict_Allfold = tf_gene_link_score_dict_Allfold,
                                    count_fold = count_fold,
                                    link_score_quantile = link_score_quantile)
    
    # delete links
    # construct total TFGene_score_dict
    TfGene_score_dict = {}
    for tf in tf_list:

        idx = gene_list.index(tf)
        idx_list = list(np.arange(len(gene_list)))
        idx_list.remove(idx)

        # count the show times of folds and construct the tf_recover_link
        tf_folds_array = [tmp[tf] for key,tmp in tf_gene_link_dict_Allfold.items()]
        tf_folds_array = np.array(tf_folds_array)
        tf_folds_score_array = np.array([tmp[tf] for key,tmp in tf_gene_link_score_dict_Allfold.items()])
        count = np.sum(tf_folds_array,axis=0)
        link_score = np.mean(tf_folds_score_array,axis=0)

        for i,gene in enumerate(np.array(gene_list)[np.array(idx_list)]):
            TfGene_score_dict['_'.join([tf,gene])] = link_score[i]

        # # filter the links with fewer links across folds
        # tf_waiting_list = np.array(gene_list)[np.array(idx_list)[count>=count_fold]]
        # link_score_filter = link_score[count>=count_fold]

    # get the values of filtered_links
    df_tmp = links_part.filtered_links[combine]
    filter_score_list = []
    for i in range(threshold_number):
        filter_score_list.append(TfGene_score_dict['_'.join([df_tmp.iloc[i,0],df_tmp.iloc[i,1]])])

    # get the thre
    thre = np.quantile(filter_score_list,delete_percent)
    
    # get rows that are less than thre
    filter_score_list = np.array(filter_score_list)
    delete_row_list = []
    for i in range(threshold_number):
        if TfGene_score_dict['_'.join([df_tmp.iloc[i,0],df_tmp.iloc[i,1]])] < thre:
            delete_row_list.append(i)
            
    df_tmp = df_tmp.reset_index(drop=True)
    df_tmp.drop(delete_row_list)
    # links_part_filter = links_part.copy()
    links_part_filter = deepcopy(links_part)
    links_part_filter.filtered_links[combine] = df_tmp.drop(delete_row_list)
    
    # Add links
    # add the links to a new link object and redo the GRN inference
    links_recover, tf_recover_filter_link =   add_recover_links(links = links_part_filter,
                                        tf_list = tf_list,
                                        init_cluster = combine,
                                        tf_recover_link = tf_recover_link,
                                        tf_target_iter_link = tf_target_iter_link)
    # refit the GRN
    oracle_part.get_cluster_specific_TFdict_from_Links(links_object=links_recover)
    oracle_part.fit_GRN_for_simulation(alpha=alpha_fit_GRN,
                                    use_cluster_specific_TFdict=True)
    
    # update the gene_GRN_mtx and tf_GRN_mtx, then again use add_recover_links function
    # to encode tf_GRN_mtx
    gene_GRN_mtx = oracle_part.coef_matrix_per_cluster[combine].copy()
    tf_GRN_mtx = gene_GRN_mtx[~(gene_GRN_mtx == 0).all(axis=1)]
    
    # update the links_recover
    links_recover, tf_recover_filter_link =   add_recover_links(links = links_part_filter,
                                        tf_list = tf_list,
                                        init_cluster = combine,
                                        tf_recover_link = tf_recover_link,
                                        tf_target_iter_link = tf_target_iter_link,
                                        tf_GRN_mtx = tf_GRN_mtx)
    
    
    # reload the parameter
    gene_GRN_mtx, tf_GRN_mtx, tf_GRN_dict = get_GRN_parameters(oracle_part, combine)
    
    
    return gene_GRN_mtx, tf_GRN_mtx, tf_GRN_dict, links_recover
