from sklearn.model_selection import KFold
import pickle
import os
import dgl
import numpy as np
import random
import scipy.sparse as sp
import torch
import itertools
import tqdm
import matplotlib.pyplot as plt
from .utils import fix_seed
from .GNN_module import *
import copy

def recover_links(g,
                  gene_list = None,
                  n_splits = 10,
                  seed = 2023,
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
                  model_dir = None,
                  pred_dir = None,
                  verbose = True,
                  save_dir = None,
                  tf_list = None,
                  best_mode = True,
                  save_metric = False,):
    '''
    This function is used to train a GNN model to recover the possible links.
    We add gpu training and earlystopping, and we provide multiple parameters to adjust the model.
    
    Parameters:
    ----------
    g: dgl graphdataset. graph used for training.
    n_splits: int. the number of the folds.
    seed: int
    neg_link_split: str.
        1. 'all' use all remaining negative links to train the model
        2. 'neg_hard_sampling' use the paired links to the positive links
    model_name: str. 'GraphSAGE'/'GAT'/'GCN'
    pred_name: str. 'mlp'/'dot'
    hidden_dim: int. GNN parameter.
    out_dim: int. GNN parameter.
    num_heads: [int]. GAT heads.
    epochs: int. training parameters.
    lr: int. learning rate.
    patience: int.
    device: str. 'cuda'/'cpu'
    verbose: bool. if verbose, print the results and plot.
    save_dir: the format is 'A/B'
    
    
    Return:
    ----------

    '''
    u, v = g.edges()
    
    # set the kfold
    kf = KFold(n_splits=n_splits,shuffle=True,random_state=42)

    # Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())),shape=(g.num_nodes(),g.num_nodes()))
    adj_neg = 1 - adj.todense() - np.eye(g.num_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    # initial parameters to save 10fold instance
    tf_gene_link_dict_Allfold = {}
    tf_gene_link_score_dict_Allfold = {}
    model_dict_Allfold = {}
    h_dict_Allfold = {}
    thre_dict_Allfold = {}

    # fix seed
    fix_seed(seed)
    dgl.seed(seed)

    metric_list = []
    for i, (train_index, test_index) in enumerate(kf.split(np.arange(g.num_edges()))):
        # if i!=9:continue
        print(f'fold {i}')
        # print(train_index.shape, test_index.shape)

        # get the links
        test_pos_u, test_pos_v = u[test_index], v[test_index]
        train_pos_u, train_pos_v = u[train_index], v[train_index]

        # split the negative links
        if neg_link_split=='all':
            neg_eids = np.random.permutation(np.arange(len(neg_u)))
            test_size = len(test_index)
            test_neg_u, test_neg_v = (
                neg_u[neg_eids[:test_size]],
                neg_v[neg_eids[:test_size]],
            )
            train_neg_u, train_neg_v = (
                neg_u[neg_eids[test_size:]],
                neg_v[neg_eids[test_size:]],
            )
        elif neg_link_split=='neg_hard_sampling':
            #TODO
            train_neg_u, train_neg_v = [], []
            # count the links for each tf
            tmp_dict = dict(Counter(train_pos_u.numpy()))
            for tf_idx,tmp_count in tmp_dict.items():
                # from all the neg samples to sample the hard negative samples
                neg_samples = list(zip(neg_u[neg_u==tf_idx],neg_v[neg_u==tf_idx]))
                train_neg_samples = random.sample(neg_samples,k=tmp_count)
                train_neg_u_tmp, train_neg_v_tmp = zip(*neg_samples)
                train_neg_u_tmp, train_neg_v_tmp = list(train_neg_u_tmp), list(train_neg_v_tmp)

                train_neg_u = train_neg_u+train_neg_u_tmp
                train_neg_v = train_neg_v+train_neg_v_tmp
                # break

            # add the train negative part to get the test negative samples
            pos_u, pos_v = list(u.numpy()), list(v.numpy())
            adj = sp.coo_matrix((np.ones(len(pos_u+train_neg_u)), (pos_u+train_neg_u, pos_v+train_neg_v)),shape=(g.num_nodes(),g.num_nodes()))
            adj_neg = 1 - adj.todense() - np.eye(g.num_nodes())
            test_neg_u, test_neg_v = np.where(adj_neg != 0)

            neg_eids = np.random.permutation(np.arange(len(test_neg_u)))
            test_size = len(test_index)
            test_neg_u, test_neg_v = (
                test_neg_u[neg_eids[:test_size]],
                test_neg_v[neg_eids[:test_size]],
            )
        else:
            raise ValueError('neg_link_split is wrong!')

        # remove the test edges
        train_g = dgl.remove_edges(g, test_index)
        if model_name=='GCN' or model_name=='GAT':
            train_g = dgl.add_self_loop(train_g)
        train_g_gpu = train_g.to(device)

        # construct the graph
        train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.num_nodes())
        train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.num_nodes())

        test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.num_nodes())
        test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.num_nodes())

        if model_name=='GraphSAGE':
            model = GraphSAGE(train_g_gpu.ndata["feat"].shape[1], out_dim).to(device)
        elif model_name=='GAT':
            model = GAT(in_size=train_g.ndata["feat"].shape[1],
                      hid_size=hidden_dim,
                      out_size=out_dim,
                      heads=num_heads).to(device)
        elif model_name=='GCN':
            model = GCN(train_g.ndata["feat"].shape[1],hidden_dim, out_dim).to(device)
        elif model_name=='MLP':
            model = MLP([train_g.ndata["feat"].shape[1], out_dim], batch_norm=False).to(device)
        else:
            raise ValueError('model_name is wrong!')


        # You can replace DotPredictor with MLPPredictor.
        if pred_name=='mlp':
            pred = MLPPredictor(out_dim).to(device)
        elif pred_name=='dot':
            pred = DotPredictor()
        else:
            raise ValueError('pred_name id wrong!')

        # set up earlystop
        early_stopping = EarlyStopping(patience=patience,
                    checkpoint_file_model=model_dir,checkpoint_file_pred=pred_dir,
                                       verbose=False)


        # in this case, loss will in training loop
        optimizer = torch.optim.Adam(
            itertools.chain(model.parameters(), pred.parameters()), lr=lr
        )

        # convert the graphs to decive
        train_pos_g_gpu = train_pos_g.to(device)
        train_neg_g_gpu = train_neg_g.to(device)
        test_pos_g_gpu = test_pos_g.to(device)
        test_neg_g_gpu = test_neg_g.to(device)
        # ----------- training -------------------------------- #

        best_metric = 0
        best_epoch = 0
        best_model = None
        best_pred = None

        # list to save results
        all_logits = []
        e_list, loss_list, eval_loss_list, auc_list, acc_list = [], [], [], [], []
        for e in tqdm.tqdm(range(epochs),total=epochs,bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            #--- train epoch
            model.train()
            pred.train()

            # forward
            if model_name == 'MLP':
                h = model(train_g_gpu.ndata["feat"])
            else:
                h = model(train_g_gpu, train_g_gpu.ndata["feat"])
            # if model_name=='GraphSAGE' or model_name=='GCN':
            #     h = model(train_g, train_g.ndata["feat"])
            # if model_name=='GAT':
            #     h = model(train_g.ndata["feat"])
            pos_score = pred(train_pos_g_gpu, h)
            neg_score = pred(train_neg_g_gpu, h)
            loss = compute_loss(pos_score, neg_score, device)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #-----eval epoch
            model.eval()
            pred.eval()
            with torch.no_grad():
                if model_name == 'MLP':
                    h = model(train_g_gpu.ndata["feat"])
                else:
                    h = model(train_g_gpu, train_g_gpu.ndata["feat"])
                pos_score = pred(test_pos_g_gpu, h)
                neg_score = pred(test_neg_g_gpu, h)
                eval_loss = compute_loss(pos_score, neg_score, device)
                thre = torch.median(torch.concat([pos_score,neg_score])).cpu().numpy()
                auc, acc, f1, precision, recall = compute_metrics(pos_score, neg_score, thre=thre)

            if best_metric < acc:
                best_metric = acc
                best_epoch = e
                best_h = h
                best_model = copy.deepcopy(model)
                best_pred = copy.deepcopy(pred)

            early_stopping(1/acc, model, pred)
            if early_stopping.early_stop:
                print(auc, acc, f1, precision, recall)
                print('EarlyStopping: run {} epoch'.format(e+1))
                break 

            if e % 10 == 0:
                # print("In epoch {}, loss: {}".format(e, loss))
                # print("test AUC", auc)
                e_list.append(e)
                loss_list.append(loss)
                eval_loss_list.append(eval_loss)
                auc_list.append(auc)
                acc_list.append(acc)

        # ----------- check results ------------------------ #
        from sklearn.metrics import roc_auc_score
        
        if best_mode:
            model = best_model
            pred = best_pred

        # calculate the metrics
        with torch.no_grad():
            # use the earlystop saving models to infer
            if model_name == 'MLP':
                h = model(train_g_gpu.ndata["feat"])
            else:
                h = model(train_g_gpu, train_g_gpu.ndata["feat"])
            pos_score = pred(test_pos_g_gpu, h)
            neg_score = pred(test_neg_g_gpu, h)
            thre = torch.median(torch.concat([pos_score,neg_score])).cpu().numpy()
            auc, acc, f1, precision, recall = compute_metrics(pos_score, neg_score, thre=thre)
            if verbose:
                print("AUC", auc)
                print('ACC', acc)
                print('F1', f1)
                print('precision', precision)
                print('recall', recall)
                print(f'the best epoch is {best_epoch}')
                print(f'the best metric is {best_metric}')
        
        metric_list.append([auc, acc, f1, precision, recall])

        # plot
        # plot the train loss and test loss
        loss_list = [l.detach().cpu().numpy() for l in loss_list]
        eval_loss_list = [l.detach().cpu().numpy() for l in eval_loss_list]
        if verbose:
            plt.rcParams["figure.figsize"] = [8, 4]
            plt.figure()
            plt.plot(e_list,loss_list)
            plt.plot(e_list,eval_loss_list)
            plt.show()

            # plot the test auc and acc
            plt.figure()
            plt.plot(e_list,auc_list)
            plt.plot(e_list,acc_list)
            plt.show()

        # construct the tf-total_target graph and predict the recovering results
        tf_gene_link_dict, tf_gene_link_score_dict = {}, {}
        for gene in tf_list:
            # gene = 'Gata3'

            # construct the test graph
            idx = gene_list.index(gene)
            idx_list = list(np.arange(g.num_nodes()))
            idx_list.remove(idx)
            tf_u = torch.Tensor([idx]*(g.num_nodes()-1)).to(torch.int32)
            tf_v = torch.Tensor(idx_list).to(torch.int32)

            # construct the graph
            tf_g = dgl.graph((tf_u, tf_v), num_nodes=g.num_nodes())

            # predict the links
            with torch.no_grad():
                if model_name == 'MLP':
                    h = model(train_g_gpu.ndata["feat"])
                else:
                    h = model(train_g_gpu, train_g_gpu.ndata["feat"])
                # h = model(train_g_gpu, train_g_gpu.ndata["feat"])
                score = pred(tf_g.to(device), h)

            # construct the link
            score_ = score.cpu().numpy()
            score_[score.cpu()>=torch.Tensor(thre)] = 1
            score_[score.cpu()<torch.Tensor(thre)] = 0
            score_ = score_.astype(int)
            tf_gene_link_dict[gene] = score_
            tf_gene_link_score_dict[gene] = score.cpu().numpy()

        tf_gene_link_dict_Allfold[str(i)] = tf_gene_link_dict
        tf_gene_link_score_dict_Allfold[str(i)] = tf_gene_link_score_dict
        model_dict_Allfold[str(i)] = pred
        h_dict_Allfold[str(i)] = h
        thre_dict_Allfold[str(i)] = thre
        
        # break
        
    if save_dir:
        with open(os.path.join(f'{save_dir}/tf_gene_link_dict_Allfold.pickle'), 'wb') as file:
            pickle.dump(tf_gene_link_dict_Allfold, file)
        with open(os.path.join(f'{save_dir}/tf_gene_link_score_dict_Allfold.pickle'), 'wb') as file:
            pickle.dump(tf_gene_link_score_dict_Allfold, file)
        # with open(os.path.join(f'{save_dir}/model_dict_Allfold.pickle'), 'wb') as file:
        #     pickle.dump(model_dict_Allfold, file)
        with open(os.path.join(f'{save_dir}/h_dict_Allfold.pickle'), 'wb') as file:
            pickle.dump(h_dict_Allfold, file)
        with open(os.path.join(f'{save_dir}/thre_dict_Allfold.pickle'), 'wb') as file:
            pickle.dump(thre_dict_Allfold, file)
    if save_metric:
        df_metric = pd.DataFrame(data = np.array(metric_list),
                                 columns = ['auc', 'acc', 'f1', 'precision', 'recall'])
        df_metric.to_csv(os.path.join(f'{save_dir}/df_metric.csv'))
            
    return tf_gene_link_dict_Allfold,tf_gene_link_score_dict_Allfold,\
model_dict_Allfold,h_dict_Allfold,thre_dict_Allfold


def filter_recover_links(tf_GRN_dict = None,
                         gene_list = None,
                         tf_list = None,
                         tf_gene_link_dict_Allfold = None,
                         tf_gene_link_score_dict_Allfold = None,
                         count_fold = 9,
                         iter_step = 1,
                         if_inv = False,
                         iter_step_inv = 4,
                         link_score_quantile = 0.1,
                         target_tf_dict = None):
    '''
   
    Parameter
    ----------
    tf_GRN_dict: dict. save the tf-target information.
    gene_list: [str].
    tf_gene_link_dict_Allfold: dict. the recovered tf-target links across all the folds.
    tf_gene_link_score_dict_Allfold: dict. the recovered tf-target link scores across all the folds.
    count_fold: int. the threshold of counting times to filter the links.
    iter_step: the propagations to filter the original links.
    if_inv: filter the links inversely linked in the original GRN.
    iter_step_inv: int.
    link_score_quantile: float 0-1. The threshold to filter the percentage of the count-based filtered links.
    
    Return
    ----------

    '''  

    
    tf_recover_link = {}
    tf_target_link = {}
    tf_target_iter_link = {}

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

        # filter the links with fewer links across folds
        tf_waiting_list = np.array(gene_list)[np.array(idx_list)[count>=count_fold]]
        link_score_filter = link_score[count>=count_fold]

        # filter the links with low predicting recover scores
        if link_score_quantile:
            if len(tf_waiting_list)==0:
                tf_recover_link[tf] = list(tf_waiting_list)
            else:
                tf_recover_link[tf] = list(tf_waiting_list[link_score_filter>=np.quantile(link_score_filter,1-link_score_quantile)])
        else:
            tf_recover_link[tf] = list(tf_waiting_list)


        # get the targets of 4 propagations
        init_gene = [tf]
        total_gene = [tf]
        waiting_gene = [tf]
        total_gene = []
        for i in range(iter_step):
            waiting_gene_tmp = []
            for gene_1 in waiting_gene:
                if gene_1 in tf_GRN_dict.keys():
                    tmp_list = []
                    for gene_2 in tf_GRN_dict[gene_1].keys():
                        if gene_2 not in total_gene:
                            tmp_list.append(gene_2)
                    total_gene = total_gene+tmp_list
                    waiting_gene_tmp = waiting_gene_tmp+tmp_list
            waiting_gene = waiting_gene_tmp

        # add targets inversely with propagations
        if if_inv:
            init_gene_inv = [tf]
            total_gene_inv = [tf]
            waiting_gene_inv = [tf]
            total_gene_inv = []
            for i in range(iter_step_inv):
                waiting_gene_tmp = []
                for gene_1 in waiting_gene_inv:
                    if gene_1 in target_tf_dict.keys():
                        tmp_list = []
                        for gene_2 in target_tf_dict[gene_1]:
                            if gene_2 not in total_gene_inv:
                                tmp_list.append(gene_2)
                        total_gene_inv = total_gene_inv+tmp_list
                        waiting_gene_tmp = waiting_gene_tmp+tmp_list
                waiting_gene = waiting_gene_tmp

            total_gene = list(set(total_gene+total_gene_inv))    

        tf_target_link[tf] = list(tf_GRN_dict[tf].keys())
        tf_target_iter_link[tf] = total_gene


    # show the recover results with dataframe
    df_tmp = {'ori_link_num':[len(tf_target_link[tf]) for tf in tf_list],
            'iter_link_num':[len(tf_target_iter_link[tf]) for tf in tf_list],
            'pred_link_num':[len(tf_recover_link[tf]) for tf in tf_list],
            'recover_link_num':[len(np.setdiff1d(tf_recover_link[tf],tf_target_iter_link[tf])) for tf in tf_list]}
    df_tmp = pd.DataFrame(df_tmp)
    df_tmp.index = tf_list
    df_tmp = df_tmp.sort_values(by='recover_link_num',ascending=False)

    return df_tmp, tf_target_link, tf_target_iter_link, tf_recover_link




def add_recover_links(links = None,
                      tf_list = None,
                      init_cluster = None,
                      tf_recover_link = None,
                      tf_target_iter_link = None,
                      tf_GRN_mtx = None):
    '''
    add the filtered recovered link to original links to create a new links object
   
    Parameter
    ----------
      
    Return
    ----------

    '''
    source_list, target_list = [], []
    coef_abs_list, coef_mean_list = [], []
    tf_recover_filter_link = {}
    
    # if tf_GRN_mtx is given, the coef_mean and the coef_abs will be write into the links object
    if not isinstance(tf_GRN_mtx,pd.DataFrame):
        for tf in tf_list:
            tmp_list = list(np.setdiff1d(tf_recover_link[tf],tf_target_iter_link[tf]))
            source_list = source_list + [tf]*len(tmp_list)
            target_list = target_list + tmp_list
            tf_recover_filter_link[tf] = tmp_list

        df_append = pd.DataFrame({'source':source_list,
                                  'target':target_list,
                                  'coef_mean':[0]*len(source_list),
                                  'coef_abs':[0]*len(source_list),
                                  'p':[0]*len(source_list),
                                  '-logp':[0]*len(source_list)})
    else:
        for tf in tf_list:
            tmp_list = list(np.setdiff1d(tf_recover_link[tf],tf_target_iter_link[tf]))
            source_list = source_list + [tf]*len(tmp_list)
            target_list = target_list + tmp_list
            for target in tmp_list:
                coef_abs_list.append(abs(tf_GRN_mtx.loc[tf,target]))
                coef_mean_list.append(tf_GRN_mtx.loc[tf,target])
            tf_recover_filter_link[tf] = tmp_list

        df_append = pd.DataFrame({'source':source_list,
                                  'target':target_list,
                                  'coef_mean':coef_mean_list,
                                  'coef_abs':coef_abs_list,
                                  'p':[0]*len(source_list),
                                  '-logp':[0]*len(source_list)})

    from copy import deepcopy
    links_recover = deepcopy(links)
    links_recover.filtered_links[init_cluster] = pd.concat([links.filtered_links[init_cluster], df_append], axis=0, ignore_index=True)
    
    return links_recover, tf_recover_filter_link