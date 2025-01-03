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