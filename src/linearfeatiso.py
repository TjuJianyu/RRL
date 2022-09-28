from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn

import torch.optim

import numpy as np 
from torch.utils.data import TensorDataset



# copyed fromhttps://github.com/moskomule/cca.pytorch/blob/master/cca/cca.py
def zero_mean(tensor: torch.Tensor, dim):
    return tensor - tensor.mean(dim=dim, keepdim=True)

# modified from https://github.com/moskomule/cca.pytorch/blob/master/cca/cca.py
def svd_reduction(tensor: torch.Tensor, accept_rate=0.99):
    left, diag, right = torch.svd(tensor)
    full = diag.abs().sum()
    ratio = diag.abs().cumsum(dim=0) / full
    num = torch.where(ratio < accept_rate,
                      torch.ones(1).to(ratio.device),
                      torch.zeros(1).to(ratio.device)
                      ).sum()
    return tensor @ right[:, :int(num)], right[:, :int(num)]

# modified from https://github.com/moskomule/cca.pytorch/blob/master/cca/cca.py
def _svd_cca(x, y):
    u_1, s_1, v_1 = x.svd()
    u_2, s_2, v_2 = y.svd()
    uu = u_1.t() @ u_2
    try:
        u, diag, v = (uu).svd(some=False)
        # u = R^{d_1 * d_1}
        # v = R^{d_2 * d_2}
        # len(diag) = min(d_1, d_2)
    except RuntimeError as e:
        raise e
    a = v_1 @ s_1.reciprocal().diag() @ u
    b = v_2 @ s_2.reciprocal().diag() @ v
    return a, b, diag

class LinearFeatISO:
    def __init__(self, method='svcca'):
        """
        :param method: method "svcca" (default) or "cca" or "pwcca"
        """
        self.method = method
    
    def fit_transform(self, x,y):
        if self.method == 'svcca':
            x, _ = svd_reduction(x)
            y, _ = svd_reduction(y)  
        

        x = zero_mean(x, dim=0)
        y = zero_mean(y, dim=0)

        self.a, self.b, self.diag = _svd_cca(x, y)
        z1_proj = x @ self.a 
        z2_proj = y @ self.b 
        
        rank = len(self.diag)

        self.alpha = z1_proj[:, :rank].abs().sum(dim=0)
        self.alpha /= self.alpha.sum() 

        z1_proj_own = torch.clone(z1_proj)
        z1_proj_own[:,:rank] -= self.diag * z2_proj[:,:rank]
        z1_proj_own[:,:rank] *= 1 / ((1 - self.diag**2).sqrt() + 1e-8)

        z2_proj_own = torch.clone(z2_proj)
        z2_proj_own[:,:rank] -= self.diag * z1_proj[:,:rank]
        z2_proj_own[:,:rank] *= 1 / ((1 - self.diag**2).sqrt() + 1e-8)

        if self.method == 'svcca' or self.method =='cca':
            return z1_proj_own, z2_proj_own, z1_proj, z2_proj, self.diag, 1 - self.diag.mean()
        else:
            return z1_proj_own, z2_proj_own, z1_proj, z2_proj, self.diag, 1 - self.alpha @ self.diag
        
class LpISO: 
    def __init__(self, args):
        self.args = args
        
    def fit_transform(self,x,y, p=2):
        args = self.args 
        x /= np.sqrt((x**2).sum(axis=1).mean())
        y /= np.sqrt((y**2).sum(axis=1).mean())
        #print((y**2).sum(axis=1).mean())
        indim = x.shape[1]
        outdim = y.shape[1]
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        dataset = TensorDataset(x,y)
        train_loader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size, shuffle=False)
        model = nn.Linear(indim, outdim).cuda()
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            nesterov=False,
                            weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.decay_epochs, gamma=args.gamma)
    
        losses = [] 
        for i in range(args.epochs):
            for x,y in train_loader:
                x = x.cuda()
                y = y.cuda()
                y_hat = model(x)

                loss = (((torch.abs(y_hat - y) ** 2).sum(axis=1) ** (p)).sum()) / len(x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            lr_scheduler.step()
            
            with torch.no_grad():
                loss = 0
                count = 0 
                examplelosses = []
                resiual = []
                for x,y in val_loader:
                    x = x.cuda()
                    y = y.cuda()
                    y_hat = model(x)
                    
                    loss += (torch.abs(y_hat - y) ** 2).sum().item()
                    count += x.shape[0]
                    examplelosses.extend((torch.abs(y_hat - y) ** 2).sum(axis=1).cpu().numpy().tolist() )
                    resiual.append((y_hat - y))

                loss /= count 
                losses.append(loss)
                #print(loss)
                examplelosses = np.array(examplelosses)
                examplelosses = np.sort(examplelosses)
                resiual = torch.cat(resiual)

                print("%.3f, %.3f, %.3f" % (loss, examplelosses[:50].mean(), examplelosses[-50:].mean()))

        return min(losses), resiual 





# # https://github.com/moskomule/cca.pytorch/blob/master/cca/cca.py
# def pwcca_distance(x, y, method="svd"):
#     """
#     Project Weighting CCA proposed in Marcos et al. 2018
#     :param x: data matrix [data, neurons]
#     :param y: data matrix [data, neurons]
#     :param method: computational method "svd" (default) or "qr"
#     """
#     a, b, diag = _cca(x, y, method=method)
#     alpha = (x @ a).abs().sum(dim=0)
#     alpha = alpha / alpha.sum()
#     return 1 - alpha @ diag
