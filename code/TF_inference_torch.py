import torch
import torch.nn as nn
from torch.utils import data
from copy import deepcopy
import matplotlib.pyplot as plt
import tqdm
import numpy as np
from plot import *

# define a loss to mix mse and corr
def correlation_score(y_true, y_pred):
    y_true = y_true.T
    y_pred = y_pred.T
    y_true_centered = y_true - torch.mean(y_true, dim=1)[:, None]
    y_pred_centered = y_pred - torch.mean(y_pred, dim=1)[:, None]
    cov_tp = torch.sum(y_true_centered * y_pred_centered, dim=1) / (y_true.shape[1] - 1)
    var_t = torch.sum(y_true_centered ** 2, dim=1) / (y_true.shape[1] - 1)
    var_p = torch.sum(y_pred_centered ** 2, dim=1) / (y_true.shape[1] - 1)
    return cov_tp / torch.sqrt(var_t * var_p)

def correlation_loss(pred, target):
    return -torch.mean(correlation_score(target, pred))

# def mix_loss(pred, target):
#     return -0.005*torch.mean(correlation_score(target, pred))+0.99*torch.mean((pred-target)**2)

class solve_loss(nn.Module):
    def __init__(self,
                 loss_name='mse',
                 lamb1=0.005,
                 lamb2=1):
        super(solve_loss, self).__init__()
        self.loss_name = loss_name
        self.lamb1 = lamb1
        self.lamb2 = lamb2
    
    def forward(self,target,pred):
        if self.loss_name == 'mse':
            loss = torch.mean(torch.pow((target - pred), 2))
        elif self.loss_name == 'corr':
            loss = correlation_loss(target,pred)
        elif self.loss_name == 'mix':
            loss = self.lamb1*correlation_loss(target,pred)+self.lamb2*torch.mean(torch.pow((target - pred), 2))
        return loss
    
    
# define a model to solve
class linear_model(nn.Module):
    def __init__(self,
                 input_shape=None):
        super().__init__()
        self.layer = nn.Linear(input_shape, 1)
        self._initialize_weights()
    
    def _initialize_weights(self):
        self.layer.weight.data.normal_(0, 0.01)
        self.layer.bias.data.fill_(0)
        
    def forward(self, x):
        return self.layer(x)
    
def load_array(data_arrays, batch_size, is_train=True):  
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
    
# define the training process
def train_model(X,
                y,
                rr,
                loss_name='mix',
                optim_name='adam',
                lr=0.03,
                weight_decay=0.001,
                train_batch=False,
                lamb1=0.005,
                lamb2=1,
                epochs=1000,
                batch_size=10,
                plot=True,
                regression_percentile = 90,
                annot_shifts=None,
                xlabel=None,
                ylabel=None,
                save=None
                ):
    X_tmp = torch.tensor(X).to(torch.float32)
    y_tmp = torch.tensor((y)).to(torch.float32)

    net = linear_model(input_shape=X_tmp.shape[1])
    
    if loss_name == 'mse':
        loss = solve_loss('mse')
        # loss = nn.MSELoss()
    elif loss_name == 'corr':
        loss = correlation_loss
    elif loss_name == 'mix':
        loss = solve_loss('mix',lamb1,lamb2)
        # loss = mix_loss
    else:
        raise ValueError('loss_name is wrong!')

    #Step 6: Define optimization algorithm 
    # implements a stochastic gradient descent optimization method

    if optim_name == 'sgd':
        trainer = torch.optim.SGD(net.parameters(), 
                                  lr=lr,
                                  momentum =0.9,
                                  weight_decay=weight_decay
                                 )
    elif optim_name == 'adam':
        trainer = torch.optim.Adam(net.parameters(), 
                                   lr=lr, 
                                   betas=(0.9, 0.99),
                                   eps=1e-08, 
                                   weight_decay=weight_decay, 
                                   amsgrad=False)
    else:
        raise ValueError('optim_name is wrong!')
    

    data_iter = load_array((X_tmp, y_tmp), batch_size)
    
    loss_list = []
    for epoch in tqdm.tqdm(range(epochs), total=epochs):
        if train_batch:
            for X_, y_ in data_iter:
                l = loss(net(X_),y_)
                trainer.zero_grad() #sets gradients to zero
                l.backward() # back propagation
                trainer.step() # parameter update
            l = loss(net(X_tmp), y_tmp)
            loss_list.append(l.detach().numpy())
        else:
            l = loss(net(X_tmp) ,y_tmp)
            trainer.zero_grad() #sets gradients to zero
            l.backward() # back propagation
            trainer.step() # parameter update
            l = loss(net(X_tmp) ,y_tmp)
            # print(f'epoch {epoch + 1}, loss {l:f}')
            loss_list.append(l.detach().numpy())
    
    rr_corr = deepcopy(rr)
    rr_corr.coef_ = net.layer.weight.detach().numpy().ravel()
    rr_corr.intercept_ = net.layer.bias.detach().numpy().ravel()[0]
    if plot:
        plt.figure()
        plt.rcParams["figure.figsize"] = [8, 4]
        plt.plot(loss_list)
        
        plt.figure()
        plot_coef(coef=rr_corr.coef_,
                  regression_percentile=regression_percentile,
                  annot_shifts=annot_shifts,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  save=save)

    my_rho = np.corrcoef(np.array((net(X_tmp)).detach().numpy().ravel()), np.array(y_tmp).ravel())
    # print('==========model:{0}, alpha:{1}'.format(model,alpha))
    print('correlation is:',my_rho[0,1])

    return rr_corr