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
    
    
def plot_umap_transition(adata,
                         change_tf,
                         rr,
                         X,
                         cluster_name_for_GRN_unit,
                         init_cluster,
                         control_cluster,
                         init_ave,
                         control_ave,
                         pca_train,
                         umap_train,
                         mode='single',
                         layer_use='normalized_count',
                         bbox_list=None,
                         title_list=None,
                         fig_size=None,
                         save=None,
                         save_prefix=None,
                         if_close=False,
                         tf_GRN_mtx=None,
                         oracle=None):
    # calculate the shift
    shift_coef = []
    if mode=='multi':
        if len(change_tf)==1:raise ValueError('the mode is wrong!')
        for (i,gene) in enumerate(tf_GRN_mtx.index): 
            if gene in list(change_tf):
                shift_coef.append(rr.coef_[i])
            else:
                shift_coef.append(0)


    elif mode=='single':
        if len(change_tf)>1:raise ValueError('the mode is wrong!')
        TF_pcc_dict, TF_acc_dict = {}, {}
        for TF in list(change_tf):
            shift_coef = []
            for (i,gene) in enumerate(tf_GRN_mtx.index): 
                if gene==TF:
                    shift_coef.append(rr.coef_[i])
                else:
                    shift_coef.append(0)
    
    # calculate the total gene shift
    # gene_shift = tf_GRN_mtx.T.values.dot((shift_coef))+rr.intercept_
    gene_shift = X.dot((shift_coef))+rr.intercept_
    
    gem = _adata_to_df(oracle.adata, layer_use)
    
    gem_transition = gem.copy()
    gem_transition.iloc[:,:] = gem_transition.iloc[:,:]+gene_shift
    
    # filter gem_transition to the init_cluster
    cluster_info = adata.obs[cluster_name_for_GRN_unit]
    gem_transition = gem_transition[(cluster_info == init_cluster)]
    
    X_umap_list, label_list=[], []
    
    # use the trained pca and umap to show the transition
    X_pca = pca_train.transform(gem.values)
    X_umap = umap_train.transform(X_pca)
    X_umap_list.append(X_umap)
    label_list.append(adata.obs[cluster_name_for_GRN_unit])
    
    # get the transition part
    X_pca = pca_train.transform(gem_transition.values)
    X_umap = umap_train.transform(X_pca)
    X_umap_list.append(X_umap)
    label_list.append([init_cluster+' transition']*len(X_umap))
    
    # bbox_list = [1.75, 1.2]
    
    # plot the transition respectively
    fig=plt.figure(figsize=fig_size)
    for i in range(2):
        fig=plt.figure(figsize=fig_size)
        df = {'UMAP1':X_umap_list[i][:, 0],\
              'UMAP2':X_umap_list[i][:, 1], \
              'label':label_list[i]}
        df = pd.DataFrame(df)
        ax = sns.scatterplot(x="UMAP1", 
                             y="UMAP2", 
                             hue="label",
                             edgecolor='none',
                             # hue_order=celltypes,
                             # saturation=1,
                             palette = 'tab10', 
                             s=8,linewidth = 0.0001, data=df)
        plt.xticks(rotation=0,fontsize=15)
        plt.yticks(rotation=0,fontsize=15)

        # ax.set(title='original UMAP',xlabel='UMAP_1')
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        if i>0:
            ax.set_xlim(lim1_x,lim2_x)
            ax.set_ylim(lim1_y,lim2_y)
        ax.set_title(title_list[i],fontsize=18)
        axLine, axLabel = ax.get_legend_handles_labels()
        # ax.legend([],[],frameon=False)
        ax.legend(loc='upper right',bbox_to_anchor=(bbox_list[i], 1),
                 frameon=False)
        if i==0:
            lim1_y,lim1_x,lim2_y,lim2_x=ax.get_ylim()[0],ax.get_xlim()[0],ax.get_ylim()[1],ax.get_xlim()[1]
        if save:
            if mode=='multi':
                fig.savefig(os.path.join(save,f'tansition visualization-{mode}-{save_prefix}-{title_list[i]}.png'),facecolor='white',bbox_inches='tight',dpi=400)
            elif mode=='single':
                fig.savefig(os.path.join(save,f'tansition visualization-{mode}-{change_tf[0]}-{save_prefix}-{title_list[i]}.png'),facecolor='white',bbox_inches='tight',dpi=400)
        if if_close:plt.close()
    # fig.savefig(save,facecolor='white',bbox_inches='tight',dpi=400)
    
    
    # get the centroids of init, control, transition
    init_centroid = umap_train.transform(pca_train.transform(init_ave.reshape(1,-1)))
    control_centroid = umap_train.transform(pca_train.transform(control_ave.reshape(1,-1)))
    transition_centroid = umap_train.transform(pca_train.transform((init_ave.ravel()+gene_shift).reshape(1,-1)))

    x_init, y_init = init_centroid[0][0],init_centroid[0][1]
    x_control, y_control = control_centroid[0][0],control_centroid[0][1]
    x_transition, y_transition = transition_centroid[0][0],transition_centroid[0][1]
    
    # plot the transition with arrow
    for i in range(1):
        fig=plt.figure(figsize=fig_size)
        df = {'UMAP1':list(X_umap_list[0][adata.obs[cluster_name_for_GRN_unit].isin([init_cluster,control_cluster])][:, 0])+list(X_umap_list[1][:, 0]),\
              'UMAP2':list(X_umap_list[0][adata.obs[cluster_name_for_GRN_unit].isin([init_cluster,control_cluster])][:, 1])+list(X_umap_list[1][:, 1]), \
              'label':list(adata.obs[adata.obs[cluster_name_for_GRN_unit].isin([init_cluster,control_cluster])][cluster_name_for_GRN_unit])+list(label_list[1])}
        df = pd.DataFrame(df)
        ax = sns.scatterplot(x="UMAP1", 
                             y="UMAP2", 
                             hue="label",
                             edgecolor='none',
                             # hue_order=celltypes,
                             # saturation=1,
                             palette = 'tab10', 
                             s=8,linewidth = 0.0001, data=df)

        size = 50
        marker = 'x'
        c = 'black'
        arrow_scale = 0.9
        plt.scatter(x_init, y_init,c=c,s=size,marker=marker)
        plt.scatter(x_control, y_control,c=c,s=size,marker=marker)
        plt.scatter(x_transition, y_transition,c=c,s=size,marker=marker)
        plt.arrow(x_init+(x_control-x_init)*(1-arrow_scale), y_init+(y_control-y_init)*(1-arrow_scale),(x_control-x_init)*(2*arrow_scale-1), (y_control-y_init)*(2*arrow_scale-1),color='black',width=0.05,shape='full')
        plt.arrow(x_init+(x_transition-x_init)*(1-arrow_scale), y_init+(y_transition-y_init)*(1-arrow_scale),(x_transition-x_init)*(2*arrow_scale-1), (y_transition-y_init)*(2*arrow_scale-1),color='red',width=0.05,shape='full')

        plt.xticks(rotation=0,fontsize=15)
        plt.yticks(rotation=0,fontsize=15)

        # ax.annotate("123", xy=(2, 2), xytext=(1, 1),
        #         arrowprops=dict(arrowstyle="->"))

        # ax.set(title='original UMAP',xlabel='UMAP_1')
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        if i>0:
            ax.set_xlim(lim1_x,lim2_x)
            ax.set_ylim(lim1_y,lim2_y)
        ax.set_title(title_list[2],fontsize=18)
        axLine, axLabel = ax.get_legend_handles_labels()
        # ax.legend([],[],frameon=False)
        ax.legend(loc='upper right',bbox_to_anchor=(bbox_list[2], 1),
                 frameon=False)
        if i==0:
            lim1_y,lim1_x,lim2_y,lim2_x=ax.get_ylim()[0],ax.get_xlim()[0],ax.get_ylim()[1],ax.get_xlim()[1]
        if save:
            if mode=='multi':
                fig.savefig(os.path.join(save,f'tansition visualization-{mode}-{save_prefix}-{title_list[2]}.png'),facecolor='white',bbox_inches='tight',dpi=400)
            elif mode=='single':
                fig.savefig(os.path.join(save,f'tansition visualization-{mode}-{change_tf[0]}-{save_prefix}-{title_list[2]}.png'),facecolor='white',bbox_inches='tight',dpi=400)
        if if_close:plt.close()
        
        
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
    Make a scatter plot that shows the relationship of a specific network score in two groups.

    Args:
        links (Links object): See network_analisis.Links class for detail.
        value (srt): The network score to be shown.
        cluster1 (str): Cluster nome to analyze. Network scores in the cluste1 are shown as x-axis.
        cluster2 (str): Cluster nome to analyze. Network scores in the cluste2 are shown as y-axis.
        percentile (float): Genes with a network score above the percentile will be shown with annotation. Default is 99.
        annot_shifts ((float, float)): Shift x and y cordinate for annotations.
        save (str): Folder path to save plots. If the folde does not exist in the path, the function create the folder.
            If None plots will not be saved. Default is None.
        select (str):
            1.'union'
            2.'intersect'
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


def GSEA_plot(adata,
              init_cluster,
              control_cluster,
              cluster_name_for_GRN_unit,
              total_tf_GRN_dict,
              save=None,
              if_dotplot=True,
              if_gsea_plot=True,
              change_tf = None
              ):
    # get the adata of init_cluster and control_cluster
    adata_part = adata[adata.obs[cluster_name_for_GRN_unit].isin([init_cluster, control_cluster])]
    
    # create the gene exp input
    df = pd.DataFrame(data=adata_part.X.toarray(),index=adata_part.obs_names,columns=adata_part.var_names)
    df = df.T
    df.index.name = 'Gene'

    # create class vector
    class_vector = list(adata_part.obs[cluster_name_for_GRN_unit])
    
    # create gene set
    tf_genes_pos_dict, tf_genes_neg_dict = {}, {}
    for tf in change_tf:
        df2 = pd.DataFrame.from_dict(total_tf_GRN_dict[init_cluster][tf],orient='index',columns=['value'])
        tf_genes_pos_dict[tf] = list(df2[df2.value>0].index)
        tf_genes_neg_dict[tf] = list(df2[df2.value<0].index)
    
    # gsea
    title_list = ['TF pos-regulate','TF neg-regulate']
    gene_set_list = [tf_genes_pos_dict,tf_genes_neg_dict]
    df_list = []
    for i in range(2):
        gs_res = gp.gsea(data=df, # or data='./P53_resampling_data.txt'
                         gene_sets=gene_set_list[i], # or enrichr library names
                         cls= class_vector, # cls=class_vector
                         # set permutation_type to phenotype if samples >=15
                         permutation_type='phenotype',
                         permutation_num=1000, # reduce number to speed up test
                         outdir=None,  # do not write output to disk
                         method='signal_to_noise',
                         threads=4, 
                         seed= 7,
                         min_size=1)
        
        df2 = gs_res.res2d
        df2['NES_abs'] = [abs(i) for i in list(df2.NES)]
        df_list.append(df2)
        
        if if_dotplot:
            # to save your figure, make sure that ``ofname`` is not None
            ax = dotplot(df2,
                         # y='NES',
                         # y_order=['E2f1'],
                         column="NES_abs",
                         title=title_list[i],
                         cmap=plt.cm.viridis,
                         size=3,
                         figsize=(4,6), 
                         cutoff=5,
                         ofname=os.path.join(save,f'dotplot_{title_list[i]}.png'),
                         top_term=100)
        if if_gsea_plot:
            # save each gsea figure
            terms = gs_res.res2d.Term
            for j in range(len(terms)):
                print(f'{title_list[i]} plot {terms[j]} gsea')
                gseaplot(gs_res.ranking, term=terms[j], **gs_res.results[terms[j]],ofname=os.path.join(save,f'gseaplot_{terms[j]}_{title_list[i]}.png'))
            
    return df_list[0],df_list[1]