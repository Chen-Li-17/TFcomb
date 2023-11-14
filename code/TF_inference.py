import numpy as np
import pandas as pd
from trajectory.oracle_utility import _adata_to_df
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from plot import *
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Binarizer


def control_TF_infer_recover(adata,
                    cluster_name_for_GRN_unit,
                    tf_list,
                    gene_GRN_mtx,
                    tf_GRN_mtx,
                    init_cluster,
                    control_cluster,
                    gene_GRN_mtx_ori=None ,
                    tf_GRN_mtx_ori=None,
                    prop_mode = 'soft',
                    layer_use='normalized_count',
                    model='ridge', 
                    alpha=1, 
                    plot=True,
                    a1=0.8,
                    a2=0.15,
                    a3=0.05,
                    regression_percentile=90,
                    annot_shifts=None,
                    xlabel=None,
                    ylabel=None,
                    save=None):
    '''
    A function used to infer the TFs' variation from the initial state to the control state.
    
    Parameters
    ----------
    adata: anndata.
    cluster_name_for_GRN_unit: str. the column of init_cluster and control_cluster.
    tf_list: list of TFs. Used to filter the rows of GRN matrix.
    gene_GRN_mtx: df. Rows and columns are both the total genes.
    tf_GRN_mtx: df. Rows are TFs and columns are genes.
    prop_mode: str
        1. 'fix': the recovered link without links with more than 1 propagations;
        2. 'soft': the recovered links can work as the normal links.
    init_cluster,control_cluster: str. 
    layer_use: str. the layer of adata for infering.
    model: str. regression model.
    alpha: the penalty of ridge/lasso
    plot: Bool. whether plot.
    a1,a1,a3: the coefficients for the GRN of 1st/2nd/3rd propagation.
    regression_percentile: int. the percentile to show the TFs.
    annot_shifts: (int,int). 
    xlabel,ylabel,save: strs. for plot and save figure.
    
    Return
    ----------
    rr: the trained regression model
    X: np.array. The X for training the model.
    init_ave: np.array. the average exp of init_cluster.
    control_ave: np.array. the average exp of control_cluster.
    '''
    # get the average of init_cluster and control_cluster
    # get the whole imputed count
    gem = _adata_to_df(adata, layer_use)

    # get the average expression for each cluster
    cluster_info = adata.obs[cluster_name_for_GRN_unit]
    cells_in_the_cluster_bool = (cluster_info == init_cluster)
    init_ave = np.mean(gem[cells_in_the_cluster_bool].values, axis=0).reshape(-1,1)
    cells_in_the_cluster_bool = (cluster_info == control_cluster)
    control_ave = np.mean(gem[cells_in_the_cluster_bool].values, axis=0).reshape(-1,1)

    y = (control_ave-init_ave).ravel()
    
    if model == 'ridge':
        rr = Ridge(alpha=alpha)
    elif model == 'linear':
        rr = LinearRegression()
    elif model == 'lasso':
        rr = Lasso(alpha=alpha)
    
    # get the X matrix
    X_tf, X_gene = tf_GRN_mtx.values.T, gene_GRN_mtx.values.T
    X_tf_ori, X_gene_ori = tf_GRN_mtx_ori.values.T, gene_GRN_mtx_ori.values.T
    if prop_mode == 'fix':
        X = a1*(X_tf)+a2*(X_gene_ori.dot(X_tf_ori))+a3*(X_gene_ori.dot(X_gene_ori).dot(X_tf_ori))
    elif prop_mode == 'soft':
        X = a1*(X_tf)+a2*(X_gene.dot(X_tf))+a3*(X_gene.dot(X_gene).dot(X_tf))
    # X = X_tf
    
    # fit model and calculate the PCC
    rr.fit(X,y)
    my_rho = np.corrcoef(rr.predict(X), y)
    print('==========model:{0}, alpha:{1}'.format(model,alpha))
    print('correlation is:',my_rho[0,1])
    
    coef = rr.coef_.ravel()
    if plot:
        plot_coef(coef=coef,
                  regression_percentile=regression_percentile,
                  annot_shifts=annot_shifts,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  save=save,
                  tf_list=tf_list)
        
    return rr,X,init_ave,control_ave

def control_TF_infer(adata,
                    cluster_name_for_GRN_unit,
                    tf_list,
                    gene_GRN_mtx,
                    tf_GRN_mtx,
                    init_cluster,
                    control_cluster,
                    layer_use='normalized_count',
                    model='ridge', 
                    alpha=1, 
                    plot=True,
                    a1=0.8,
                    a2=0.15,
                    a3=0.05,
                    regression_percentile=90,
                    annot_shifts=None,
                    xlabel=None,
                    ylabel=None,
                    save=None):
    '''
    A function used to infer the TFs' variation from the initial state to the control state.
    
    Parameters
    ----------
    adata: anndata.
    cluster_name_for_GRN_unit: str. the column of init_cluster and control_cluster.
    tf_list: list of TFs. Used to filter the rows of GRN matrix.
    gene_GRN_mtx: df. Rows and columns are both the total genes.
    tf_GRN_mtx: df. Rows are TFs and columns are genes.
    init_cluster,control_cluster: str. 
    layer_use: str. the layer of adata for infering.
    model: str. regression model.
    alpha: the penalty of ridge/lasso
    plot: Bool. whether plot.
    a1,a1,a3: the coefficients for the GRN of 1st/2nd/3rd propagation.
    regression_percentile: int. the percentile to show the TFs.
    annot_shifts: (int,int). 
    xlabel,ylabel,save: strs. for plot and save figure.
    
    Return
    ----------
    rr: the trained regression model
    X: np.array. The X for training the model.
    init_ave: np.array. the average exp of init_cluster.
    control_ave: np.array. the average exp of control_cluster.
    '''
    # get the average of init_cluster and control_cluster
    # get the whole imputed count
    gem = _adata_to_df(adata, layer_use)

    # get the average expression for each cluster
    cluster_info = adata.obs[cluster_name_for_GRN_unit]
    cells_in_the_cluster_bool = (cluster_info == init_cluster)
    init_ave = np.mean(gem[cells_in_the_cluster_bool].values, axis=0).reshape(-1,1)
    cells_in_the_cluster_bool = (cluster_info == control_cluster)
    control_ave = np.mean(gem[cells_in_the_cluster_bool].values, axis=0).reshape(-1,1)

    y = (control_ave-init_ave).ravel()
    
    if model == 'ridge':
        rr = Ridge(alpha=alpha)
    elif model == 'linear':
        rr = LinearRegression()
    elif model == 'lasso':
        rr = Lasso(alpha=alpha)
    
    # get the X matrix
    X_tf, X_gene = tf_GRN_mtx.values.T, gene_GRN_mtx.values.T
    X = a1*(X_tf)+a2*(X_gene.dot(X_tf))+a3*(X_gene.dot(X_gene).dot(X_tf))
    
    
    # fit model and calculate the PCC
    rr.fit(X,y)
    my_rho = np.corrcoef(rr.predict(X), y)
    print('==========model:{0}, alpha:{1}'.format(model,alpha))
    print('correlation is:',my_rho[0,1])
    
    coef = rr.coef_.ravel()
    if plot:
        plot_coef(coef=coef,
                  regression_percentile=regression_percentile,
                  annot_shifts=annot_shifts,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  save=save)
        
    return rr,X,init_ave,control_ave

def cal_control_pcc(change_tf,
                    rr,
                    tf_GRN_mtx,
                    diff_ave,
                    mode='single',
                    if_print=True,
                    tf_GRN_dict=None,
                    X=None):
    '''
    given the list of TFs we want to change, and calculate the pcc of predicted change values and real change values.
    
    Parameter
    ----------
    change_tf: the list of the TFs we want to change.
    rr: the trained regression model
    tf_GRN_mtx: pandas dataframe. Rows are the total genes, columns are the regulatory TFs.
    diff_ave: the true difference between the control state and the initial state.
    mode: 
        1.'single': get the pcc of each single TF
        2.'multi': get the pcc by change all the input TFs
        
    Return
    ----------
    1.'single': return a dict of the pcc for each TF
    2.'multi': return a pcc value
    '''
    
    # change_tf = cluster_tf_score_df[cluster_tf_score_df.coef_pvalue<5e-2]
    shift_coef = []
    if mode=='multi':
        for (i,gene) in enumerate(tf_GRN_mtx.index): 
            if gene in list(change_tf):
                shift_coef.append(rr.coef_[i])
            else:
                shift_coef.append(0)
        if X is not None:
            my_rho = np.corrcoef(X.dot((shift_coef))+rr.intercept_, diff_ave)
        else:
            my_rho = np.corrcoef(tf_GRN_mtx.T.values.dot((shift_coef))+rr.intercept_, diff_ave)
        if if_print:
            print('the significant TFs change, and the correlation is:',my_rho[0,1])
        return my_rho[0,1]
    elif mode=='single':
        TF_pcc_dict, TF_acc_dict = {}, {}
        for TF in list(change_tf):
            shift_coef = []
            for (i,gene) in enumerate(tf_GRN_mtx.index): 
                if gene==TF:
                    shift_coef.append(rr.coef_[i])
                else:
                    shift_coef.append(0)

            if X is not None:
                my_rho = np.corrcoef(X.dot((shift_coef))+rr.intercept_, diff_ave)
            else:
                my_rho = np.corrcoef(tf_GRN_mtx.T.values.dot((shift_coef))+rr.intercept_, diff_ave)
            
            # calculate the accuracy
            array1, array2 = tf_GRN_mtx.T.values.dot((shift_coef))+rr.intercept_, diff_ave
            idx_list = []
            for (i,gene) in enumerate(tf_GRN_mtx.columns):
                if gene in tf_GRN_dict[TF].keys():
                    idx_list.append(True)
                else:
                    idx_list.append(False)
            array1, array2 = array1[idx_list], array2[idx_list]
            
            # my_rho = np.corrcoef(array1,array2)
            
            array1 = Binarizer().fit_transform(array1.reshape(-1,1))
            array2 = Binarizer().fit_transform(array2.reshape(-1,1))
            acc = accuracy_score(array1, array2)
            
            if if_print:
                print('the significant TF {0} change, and the correlation is:'.format(TF),my_rho[0,1])
                print('the significant TF {0} change, and ACC is:'.format(TF),acc)
            TF_pcc_dict[TF], TF_acc_dict[TF] = my_rho[0,1], acc
        return TF_pcc_dict, TF_acc_dict
    else:
        raise ValueError('wrong mode!')