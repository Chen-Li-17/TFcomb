import numpy as np
import pandas as pd
from TFcomb.tools.utils_celloracle import _adata_to_df
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from TFcomb.plotting.plot import *
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Binarizer


def TF_inference(adata,
                    cluster_name_for_GRN_unit,
                    tf_list,
                    tf_GRN_mtx,
                    gene_GRN_mtx,
                    tf_GRN_mtx_ori,
                    gene_GRN_mtx_ori,
                    init_cluster,
                    control_cluster,
                    prop_mode = 'soft',
                    layer_use='normalized_count',
                    model='ridge', 
                    alpha=1, 
                    plot=True,
                    a1=0.8,
                    a2=0.15,
                    a3=0.05,
                    regression_percentile=90,
                    annot_shifts=(2,0.01),
                    xlabel='index of TFs',
                    ylabel='Expected alteration',
                    save=None):
    """
    Infers transcription factor (TF) variation between an initial and control state.

    This function performs regression analysis using GRN matrices and imputed gene 
    expression data to infer the expected TF activity alterations between two cell states.

    Args:
        adata (anndata.AnnData): Input AnnData object containing single-cell data.
        cluster_name_for_GRN_unit (str): Column in `adata.obs` specifying cluster labels.
        tf_list (list): List of transcription factors to filter rows of the GRN matrix.
        tf_GRN_mtx (pd.DataFrame): GRN matrix with TFs as rows and genes as columns.
        gene_GRN_mtx (pd.DataFrame): GRN matrix with all genes as rows and columns.
        tf_GRN_mtx_ori (pd.DataFrame): Original GRN matrix with TFs as rows and genes as columns.
        gene_GRN_mtx_ori (pd.DataFrame): Original GRN matrix with all genes as rows and columns.
        init_cluster (str): Name of the initial cluster/state.
        control_cluster (str): Name of the control cluster/state.
        prop_mode (str, optional): Mode of propagation:
            - `'fix'`: Considers only direct links.
            - `'soft'`: Includes propagated links. Defaults to `'soft'`.
        layer_use (str, optional): Layer in `adata` to use for gene expression. Defaults to `'normalized_count'`.
        model (str, optional): Regression model to use (`'ridge'`, `'linear'`, `'lasso'`). Defaults to `'ridge'`.
        alpha (float, optional): Regularization strength for Ridge/Lasso regression. Defaults to 1.
        plot (bool, optional): Whether to generate a plot of the results. Defaults to True.
        a1 (float, optional): Weight for the first propagation level. Defaults to 0.8.
        a2 (float, optional): Weight for the second propagation level. Defaults to 0.15.
        a3 (float, optional): Weight for the third propagation level. Defaults to 0.05.
        regression_percentile (int, optional): Percentile of regression coefficients to highlight in the plot. Defaults to 90.
        annot_shifts (tuple, optional): Annotation shifts for the plot (x, y). Defaults to (2, 0.01).
        xlabel (str, optional): Label for the x-axis of the plot. Defaults to `'index of TFs'`.
        ylabel (str, optional): Label for the y-axis of the plot. Defaults to `'Expected alteration'`.
        save (str, optional): Path to save the plot. If None, the plot is not saved. Defaults to None.

    Returns:
        tuple:
            - rr (sklearn.base.RegressorMixin): Trained regression model.
            - X (numpy.ndarray): Design matrix used for regression.
            - init_ave (numpy.ndarray): Average gene expression in the initial cluster.
            - control_ave (numpy.ndarray): Average gene expression in the control cluster.
    """
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

def get_directing_score(
                    change_tf,
                    rr,
                    tf_GRN_mtx,
                    tf_GRN_dict,
                    diff_ave = None,
                    mode = 'single',
                    if_print = False,
                    X = None):
    """
    Computes the predictive correlation (PCC) and accuracy of transcription factor (TF) changes.

    This function evaluates the correlation and/or accuracy of predicted changes in target gene expression 
    based on TF alterations, using a trained regression model and the GRN matrix.

    Args:
        change_tf (list): List of TFs to modify.
        rr (sklearn.base.RegressorMixin): Trained regression model.
        tf_GRN_mtx (pd.DataFrame): GRN matrix where rows are genes and columns are TFs.
        tf_GRN_dict (dict): Dictionary mapping each TF to its regulated genes and their scores.
        diff_ave (numpy.ndarray, optional): Observed differences between control and initial states. Defaults to None.
        mode (str, optional): Evaluation mode:
            - `'single'`: Compute PCC and accuracy for each TF individually.
            - `'multi'`: Compute PCC for all input TFs together. Defaults to `'single'`.
        if_print (bool, optional): Whether to print the results. Defaults to False.
        X (numpy.ndarray, optional): Optional design matrix for regression. If None, it is calculated from `tf_GRN_mtx`. Defaults to None.

    Returns:
        Union[dict, float]: 
            - If `mode='single'`: A tuple containing:
                - `TF_pcc_dict` (dict): Dictionary mapping each TF to its PCC.
                - `TF_acc_dict` (dict): Dictionary mapping each TF to its accuracy score.
            - If `mode='multi'`: A single PCC value.
    """
    
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