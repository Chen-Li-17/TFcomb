import os
os.environ["DGLBACKEND"] = "pytorch"
from dgl.data import DGLDataset
import dgl
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


def get_edges_data(tf_GRN_mtx):
    Src, Dst, Weight = [], [], []
    gene_list = list(tf_GRN_mtx.columns)
    for i,index_ in enumerate(tf_GRN_mtx.index):
        for j,column_ in enumerate(tf_GRN_mtx.columns):
            if tf_GRN_mtx.iloc[i,j]!=0:
                Src.append(gene_list.index(index_))
                Dst.append(gene_list.index(column_))
                Weight.append(tf_GRN_mtx.iloc[i,j])
    df = pd.DataFrame({'Src':Src,
                       'Dst':Dst,
                       'Weight':Weight})
    return df

class GRN_Dataset(DGLDataset):
    def __init__(self,adata_part,tf_GRN_mtx,tf_list):
        self.adata_part = adata_part
        self.tf_GRN_mtx = tf_GRN_mtx
        self.tf_list = tf_list
        super().__init__(name="GRN")
        # self.adata_part = adata_part
        # self.tf_GRN_mtx = tf_GRN_mtx
        # self.tf_list = tf_list

    def process(self):
        
        node_features = torch.from_numpy(self.adata_part.X.T)
        node_labels = torch.from_numpy(np.array([1 if i in self.tf_list else 0 for i in self.tf_GRN_mtx.columns]))
        edges_data = get_edges_data(self.tf_GRN_mtx)
        edge_features = torch.from_numpy(edges_data["Weight"].to_numpy())
        edges_src = torch.from_numpy(edges_data["Src"].to_numpy())
        edges_dst = torch.from_numpy(edges_data["Dst"].to_numpy())

        self.graph = dgl.graph(
            (edges_src, edges_dst), num_nodes=node_labels.shape[0]
        )
        self.graph.ndata["feat"] = node_features
        self.graph.ndata["label"] = node_labels
        self.graph.edata["weight"] = edge_features

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
    
    
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,precision_score,recall_score

def compute_metrics(pos_score, neg_score, thre=0.5):
    scores = torch.cat([pos_score, neg_score]).cpu().detach().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).cpu().numpy()
    scores_ = scores.copy()
    # thre = -1
    scores_[scores>thre]=1
    scores_[scores<=thre]=0
    return roc_auc_score(labels, scores),accuracy_score(labels, scores_),\
f1_score(labels, scores_),precision_score(labels, scores_),recall_score(labels, scores_)

def compute_loss(pos_score, neg_score, device='cpu'):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).to(device)
    return F.binary_cross_entropy_with_logits(scores, labels)

#-----models
from dgl.nn import SAGEConv
# build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, "mean")
        self.conv2 = SAGEConv(h_feats, h_feats, "mean")

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)                            
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)
    
import dgl.nn as dglnn
class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, activation=F.relu)
        )
        self.layers.append(dglnn.GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h
    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)                            
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

# attention: only write load_data in GAT, which means only GAT can use earlystoping now
class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # three-layer GAT
        self.gat_layers.append(
            dglnn.GATConv(in_size, hid_size, heads[0], activation=F.elu)
        )
        self.gat_layers.append(
            dglnn.GATConv(
                hid_size * heads[0],
                hid_size,
                heads[1],
                residual=True,
                activation=F.elu,
            )
        )
        self.gat_layers.append(
            dglnn.GATConv(
                hid_size * heads[1],
                out_size,
                heads[2],
                residual=True,
                activation=None,
            )
        )

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == 2:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        return h
    
    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)                            
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

import dgl.function as fn
class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata["score"][:, 0]
    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)                            
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)
        
class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src["h"], edges.dst["h"]], 1)
        return {"score": self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]
        
    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)                            
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

class EarlyStopping:
    """Early stops the training if loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, checkpoint_file_model='',checkpoint_file_pred=''):
        """
        Args:
            patience (int): How long to wait after last time loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.checkpoint_file_model = checkpoint_file_model
        self.checkpoint_file_pred = checkpoint_file_pred

    def __call__(self, loss, model, pred):
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model, pred)
        elif score <= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_model(self.checkpoint_file_model)
                pred.load_model(self.checkpoint_file_pred)
        else:
            self.best_score = score
            self.save_checkpoint(loss, model, pred)
            self.counter = 0

    def save_checkpoint(self, loss, model, pred):
        """
        Saves model when loss decrease
        """
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_file_model)
        torch.save(pred.state_dict(), self.checkpoint_file_pred)
        self.loss_min = loss