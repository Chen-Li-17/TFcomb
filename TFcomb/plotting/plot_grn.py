import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap

def plot_GRN(
    tf_GRN_dict = None,
    plot_tf_list = None, 
    save_dir_GRN = None,
    filter_link = True,
    figsize=(15,15),
    anno_tfs = None,
    anno_edges = None,
    show_mode = True,
    seed=2,
    scale=0.5,
    iterations=5,
    k=0.1,
    gene_color_dict = None):
    """Plot the GRN with TFs and target genes.

    Args:
        tf_GRN_dict (dict, optional): A dict where keys are TFs and values are target genes. Defaults to None.
        plot_tf_list (list, optional): A list of TFs to be plotted. Defaults to None.
        save_dir_GRN (str, optional): Path to save the figure. Defaults to None.
        filter_link (bool, optional): Whether filter the links with small values. Defaults to True.
        figsize (tuple, optional): Size of figure. Defaults to (15,15).
        anno_tfs (list, optional): Nodes to be annotated. Defaults to None.
        anno_edges (list, optional): Edges to be annotated. Defaults to None.
        show_mode (bool, optional): Whether plot the figure. Defaults to True.
        seed (int, optional): Random seed. Defaults to 2.
        scale (float, optional): The scale to show the graph. Defaults to 0.5.
        iterations (int, optional): The parameters of nx.spring_layout. Defaults to 5.
        k (float, optional): The parameters of nx.spring_layout. Defaults to 0.1.
        gene_color_dict (dict, optional): A dict where keys are genes and values are colors, which is used to give colors representing the differentially expressed degree. Defaults to None.
    """

    if isinstance(plot_tf_list, str):
        plot_tf_list = [plot_tf_list]

    # Create a directed graph
    G = nx.DiGraph()

    # Add directed edges and weights
    edges_with_weights = []
    edges_colors = []
    genes = []
    edges_colors_dict = {}
    for tf in plot_tf_list:
        genes.append(tf)
        for target in tf_GRN_dict[tf].keys():
            if filter_link:
                if abs(tf_GRN_dict[tf][target]) < 0.1:
                    continue
            # if target in intersect_list:
            #     continue
            genes.append(target) 
            edges_with_weights.append((tf,target,tf_GRN_dict[tf][target]))

            if tf_GRN_dict[tf][target] > 0:
                edges_colors_dict['_'.join([tf, target])] = '#ff8884'
            else:
                edges_colors_dict['_'.join([tf, target])] = '#9ac9db'
    genes = list(np.unique(genes))
    G.add_nodes_from(genes)
    G.add_weighted_edges_from(edges_with_weights)

    for start, end, weight in G.edges(data="weight"):
        if '_'.join([start, end]) in anno_edges:
            edges_colors.append('black')
        else:
            edges_colors.append(edges_colors_dict['_'.join([start, end])])
        
    # Calculate line thickness based on weight
    edge_widths = []
    for start, end, weight in G.edges(data="weight"):
        if '_'.join([start, end]) in anno_edges:
            edge_widths.append(3 * abs(weight)) # The multiplier can be adjusted as needed
        else:
            edge_widths.append(3 * abs(weight)) # The multiplier can be adjusted as needed

    # Calculate out-degree of each node and set node sizes
    node_out_degree = dict(G.out_degree())
    node_sizes = [150 + 5 * node_out_degree[node] for node in G.nodes]  # Base size is 200, each out-degree adds 50

    # Calculate font sizes
    # Calculate out-degrees
    out_degrees = G.out_degree()
    font_sizes = {node: np.log(200 + 300 * deg) for node, deg in out_degrees}

    # Set the color for special nodes
    node_colors = ["#BEB8DC" if node in plot_tf_list else "#E7DAD2" for node in G.nodes]
    node_colors = ["red" if node in plot_tf_list else gene_color_dict[node] for node in G.nodes]

    # Draw the network
    plt.figure(figsize=figsize)
    from networkx.drawing.nx_agraph import graphviz_layout
    pos = nx.spring_layout(G, 
                           seed=seed,
                           scale=scale,
                           iterations=iterations,
                           k=k)  # Define node layout
    pos = graphviz_layout(G)

    nx.draw(G, 
            pos, 
            with_labels=False,
            node_color=node_colors, 
            node_size = node_sizes,
            font_color="black",
            width=edge_widths, 
            edge_color=edges_colors, 
            )

    # Draw node labels based on out-degrees and adjust font sizes
    for node, (x, y) in pos.items():
        if node in anno_tfs:
            plt.text(x, y, 
                     s=node, 
                     color = 'darkred',
                     # bbox=dict(facecolor='white', edgecolor='black'), 
                     ha='center', va='center', fontsize=font_sizes[node])
        else:
            plt.text(x, y, 
                     s=node, 
                     # bbox=dict(facecolor='white', edgecolor='black'), 
                     ha='center', va='center', fontsize=font_sizes[node])

    # plt.title(f"Recovered links of {tf} on {prefix_dict[prefix]}", fontsize=12)
    plt.title(f"GRN", fontsize=12)

    if save_dir_GRN:
        plt.savefig(save_dir_GRN,facecolor='white',bbox_inches='tight')
        plt.savefig(save_dir_GRN.replace('.pdf', '.png'),facecolor='white',bbox_inches='tight', dpi=300)
    
    if show_mode:
        plt.show()
    else:
        plt.close()
        
def create_gradient_color_list(n, start_color, middle_color, end_color):
    # Create a custom color map

    # Define the color map
    colors = [start_color, middle_color, end_color]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=n)

    # Generate the color list
    gradient_color_list = [cmap(i) for i in range(cmap.N)]
    return gradient_color_list
        
def get_gene_color_dict(oracle_part,
                        source_state):
    """Generate the gene_color_dict

    Args:
        oracle_part (celloracle object): CellOracle object.
        source_state (str): Name of the source state.

    Returns:
        gene_color_dict (dict): A dict mapping genes to colors.
        gene_color_dict_2 (dict): A dict mapping genes to colors 2.
    """
    
    adata = oracle_part.adata.copy()

    sc.tl.rank_genes_groups(adata, 
                            use_raw = False,
                            groupby='celltype', 
                            reference = source_state,
                            rankby_abs = False,
                            method = 'wilcoxon'
                           )
    de_genes = pd.DataFrame(adata.uns['rank_genes_groups']['names'])
    de_genes = list(de_genes.iloc[:,0])

    # Define colors
    start_color = "#800080"  # Red
    middle_color = "white"  # Green
    end_color = "#FFD700"  # Blue

    color_list = create_gradient_color_list(len(de_genes), start_color, middle_color, end_color)
    gene_color_dict = {}
    for i, gene in enumerate(de_genes):
        gene_color_dict[gene] = color_list[i]

    # Visualize the gradient effect
    plt.figure(figsize=(10, 2))
    plt.imshow([color_list], aspect='auto')
    plt.axis('off')
    plt.title("Color Gradient from Red to Green to Blue")
    plt.show()
    
    return gene_color_dict
