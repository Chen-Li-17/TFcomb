import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns
import gseapy as gp
from gseapy import dotplot
from gseapy import gseaplot, heatmap
from TFcomb.tools.utils_celloracle import _adata_to_df
from adjustText import adjust_text
# from trajectory.oracle_utility import _adata_to_df
settings = {"save_figure_as": "png"}

def plot_scores_as_rank(df, 
                        cluster,
                        links=None,
                        values=None, 
                        n_gene=50, 
                        save=None):
    """
    Pick up top n-th genes wich high-network scores and make plots.

    Args:
        links (Links object): See network_analisis.Links class for detail.
        cluster (str): Cluster nome to analyze.
        n_gene (int): Number of genes to plot. Default is 50.
        save (str): Folder path to save plots. If the folde does not exist in the path, the function create the folder.
            If None plots will not be saved. Default is None.
    """
    if values == None:
        values = ['degree_centrality_all',
                      'degree_centrality_in', 'degree_centrality_out',
                      'betweenness_centrality',  'eigenvector_centrality']
    for value in values:

        res = df[df.cluster == cluster]
        res = res[value].sort_values(ascending=False)
        res = res[:n_gene]

        fig = plt.figure()

        plt.scatter(res.values, range(len(res)))
        plt.yticks(range(len(res)), res.index.values)#, rotation=90)
        plt.xlabel(value)
        plt.title(f" {value} \n top {n_gene} in {cluster}")
        plt.gca().invert_yaxis()
        plt.subplots_adjust(left=0.5, right=0.85)

        if not save is None:
            os.makedirs(save, exist_ok=True)
            path = os.path.join(save, f"ranked_values_in_{links.name}_{value}_{links.threshold_number}_in_{cluster}.{settings['save_figure_as']}")
            fig.savefig(path, transparent=True)
        plt.show()
        
def plot_coef(coef,
              regression_percentile,
              annot_shifts=None,
              xlabel=None,
              ylabel=None,
              save=None,
              tf_list=None):
    """Plot the coefficients of the ridge regression model. The coefficients are also regarded as the 
    expected alterations.

    Args:
        coef (list): The coefficients of regression model. The length is equal to the length of TFs.
        regression_percentile (float): The percentage to show the TFs. 
        annot_shifts (tuple, optional): A pair of coordinates. The shifts of text to the point. Defaults to None.
        xlabel (str, optional): Name of x label. Defaults to None.
        ylabel (str, optional): Name of y label. Defaults to None.
        save (str, optional): The path to save the figure. Defaults to None.
        tf_list (list, optional): List containing TFs. Defaults to None.
    """
    
    thre_up = np.percentile(coef[coef>=0],regression_percentile)
    thre_down = np.percentile(coef[coef<0],100-regression_percentile)
    
    if annot_shifts is None:
        x_shift, y_shift = (coef.max() - coef.min())*0.03, (coef.max() - coef.min())*0.03
    else:
        x_shift, y_shift = annot_shifts

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.stem(coef)
    shift_flag = 1
    for (i,coef_) in enumerate(coef):
        if coef_>thre_up or coef_<thre_down:
            # shift_flag = shift_flag*(-1)
            # plt.scatter(i, coef_, 
            #             c="r", 
            #             s=50,
            #             edgecolor="red",
            #             marker='o')
            _plot_goi(i, coef_, tf_list[i], {}, scatter=False, x_shift=shift_flag*x_shift, y_shift=shift_flag*y_shift,color='red')
    if save:
        plt.savefig(save,dpi=400)
    plt.show()
    
    
def _plot_goi(x, y, goi, args_annot, scatter=False, x_shift=0.1, y_shift=0.1, color='black', goi_size=10):
    """
    Plot an annoation to highlight one point in scatter plot.

    Args:
        x (float): Cordinate-x.
        y (float): Cordinate-y.
        args_annot (dictionary): arguments for matplotlib.pyplot.annotate().
        scatter (bool): Whether to plot dot for the point of interest.
        x_shift (float): distance between the annotation and the point of interest in the x-axis.
        y_shift (float): distance between the annotation and the point of interest in the y-axis.
    """

    default = {"size": goi_size}
    default.update(args_annot)
    args_annot = default.copy()

    arrow_dict = {"width": 0.5, "headwidth": 0.5, "headlength": 1, "color": color}
    if scatter:
        plt.scatter(x, y, c="none", edgecolor=color)
    plt.annotate(f"{goi}", xy=(x, y), 
                #  textcoords="offset points",
                #  xytext=(x+0.5, y+0.5),
                 xytext=(x+x_shift, y+y_shift),
                 color=color, arrowprops=arrow_dict, **args_annot)
    
def _plot_goi_2(x, y, goi, args_annot, scatter=False, x_shift=0.1, y_shift=0.1, color='black', goi_size=10, weight='normal'):
    """
    Plot an annoation to highlight one point in scatter plot.

    Args:
        x (float): Cordinate-x.
        y (float): Cordinate-y.
        args_annot (dictionary): arguments for matplotlib.pyplot.annotate().
        scatter (bool): Whether to plot dot for the point of interest.
        x_shift (float): distance between the annotation and the point of interest in the x-axis.
        y_shift (float): distance between the annotation and the point of interest in the y-axis.
    """
    
    default = {"size": goi_size}
    default.update(args_annot)
    args_annot = default.copy()

    arrow_dict = {"width": 0.2, "headwidth": 0.5, "headlength": 1, "color": color}
    if scatter:
        plt.scatter(x, y, c="none", edgecolor=color)
    anno = plt.annotate(f"{goi}", xy=(x, y), 
                #  textcoords="offset points",
                 xytext=(x, y),
                        weight=weight,
                 # xytext=(x+x_shift, y+y_shift),
                 color=color, 
                        # arrowprops=arrow_dict,
                        **args_annot)
    return anno
    
        
        
def plot_score_comparison(
                        df,
                        value,
                        cluster1,
                        cluster2,
                        percentile1=99,
                        percentile2=99, 
                        annot_shifts=None,
                        fillna_with_zero=True,
                        plt_show=True,
                        de_genes = [],
                        gt_tfs = [],
                        title = None,
                        if_line=True,
                        plot_all_tf = False,
                        save=None,
                        goi_size=10,
                        point_size=15,
                        ):
    """
    Create a scatter plot to visualize the directing scores and expected alterations of transcription factors (TFs).

    Args:
        df (pd.DataFrame): A Pandas DataFrame containing directing scores and expected alterations.
        value (str): The column name representing the values in `df`.
        cluster1 (str): The column name for expected alterations.
        cluster2 (str): The column name for directing scores.
        percentile1 (int): The quantile percentage for filtering TFs based on expected alterations. Defaults to 99.
        percentile2 (int, optional): The quantile percentage for filtering TFs based on directing scores. Defaults to 99.
        annot_shifts (tuple, optional): A pair of coordinates specifying the text shift relative to data points. Defaults to None.
        fillna_with_zero (bool, optional): Whether to fill missing values in `df` with zeros. Defaults to True.
        plt_show (bool, optional): Whether to display the plot. Defaults to True.
        de_genes (list, optional): A list of genes differentially expressed between source and target cell states. Defaults to an empty list.
        gt_tfs (list, optional): A list of ground-truth TFs. Defaults to an empty list.
        title (str, optional): The title of the plot. Defaults to None.
        if_line (bool, optional): Whether to display a red line indicating the filter threshold. Defaults to True.
        plot_all_tf (bool, optional): Whether to plot all TFs, regardless of filtering. Defaults to False.
        save (str, optional): The file path to save the plot. Defaults to None.
        goi_size (int, optional): The font size for gene-of-interest (GOI) labels. Defaults to 10.
        point_size (int, optional): The size of scatter plot points. Defaults to 15.

    Returns:
        gois (list): A list of filtered TFs based on the specified criteria.
    """

    piv = pd.pivot_table(df, values=value, columns="cluster", index="index")
    if fillna_with_zero:
        piv = piv.fillna(0)
    else:
        piv = piv.fillna(piv.mean(axis=0))

    goi1 = piv[piv[cluster1] > np.percentile(piv[cluster1][piv[cluster1].values>0].values, percentile1)].index
    goi2 = piv[piv[cluster1] < np.percentile(piv[cluster1][piv[cluster1].values<0].values, 100-percentile1)].index
    goi3 = piv[piv[cluster2] > np.percentile(piv[cluster2].values, percentile2)].index
    goi4 = piv[piv[cluster1] > 0].index
    # gois = np.intersect1d(goi4,np.union1d(goi1, goi3))
    gois = np.union1d(goi2,np.union1d(goi1, goi3))
    
    if plot_all_tf:
        gois = np.array(piv.index)
    
    thre1 = np.percentile(piv[cluster1][piv[cluster1].values>0].values, percentile1)
    thre2 = np.percentile(piv[cluster2].values, percentile2)
    thre3 = np.percentile(piv[cluster1][piv[cluster1].values<0].values, 100-percentile1)
    

    x, y = piv[cluster1], piv[cluster2]
    plt.scatter(x, y, c="none", edgecolor="black",s=point_size)

    if annot_shifts is None:
        x_shift, y_shift = (x.max() - x.min())*0.03, (y.max() - y.min())*0.03
    else:
        x_shift, y_shift = annot_shifts


    anno_list = []
    for goi in gois:
        x, y = piv.loc[goi, cluster1], piv.loc[goi, cluster2]
        if goi in gt_tfs:
            plt.scatter(x, y, c="none", edgecolor="black",s=point_size)
            anno = _plot_goi_2(x, y, goi, {}, scatter=False, x_shift=x_shift, y_shift=y_shift,color='r',goi_size=goi_size,weight='bold')
            anno_list.append(anno)
        else:
            if goi in de_genes:
                plt.scatter(x, y, c="none", edgecolor="black",s=point_size)
                anno = _plot_goi_2(x, y, goi, {}, scatter=False, x_shift=x_shift, y_shift=y_shift,color='#1f77b4',goi_size=goi_size,weight='bold')
                anno_list.append(anno)
            else:
                plt.scatter(x, y, c="none", edgecolor='black',s=point_size)
                anno = _plot_goi_2(x, y, goi, {}, scatter=False, x_shift=x_shift, y_shift=y_shift,color='#1f77b4',goi_size=goi_size)
                anno_list.append(anno)
    adjust_text(anno_list,lim=500,
                expand_objects=(1.05,1.05),
               force_text=(0.02,0.01),
                precision=0.005,
                arrowprops=dict(arrowstyle="-", color='k', lw=0.5)
               )        
    

    # plt.xlabel(cluster1)
    # plt.ylabel(cluster2)
    plt.xlabel('Expected alteration')
    plt.ylabel('Directing score')
    plt.axvline(x=0,color='black')
    plt.axhline(y=0,color='black')
    if if_line and not plot_all_tf:
        plt.axvline(x=thre1,color='r')
        plt.axvline(x=thre3,color='r')
        plt.axhline(y=thre2,color='r')
        # plt.axhline(y=thre3,color='r')

    xmin, xmax, ymin, ymax = plt.axis()
    if title == None:
        plt.title(f"{value}")
    else:
        plt.title(title)

    if save:
        plt.savefig(save ,bbox_inches='tight',facecolor='white')
    if plt_show:
        plt.show()
    else:
        plt.close()
            
    return gois

