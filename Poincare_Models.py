from Comp_Poincare import PoincareEnLiFu, get_activate_func
from hyptorch.nn import HNNLayer
from hyptorch.poincare.manifold import PoincareBall 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_scipy_sparse_matrix
import numpy as np
class Poincare_MH(nn.Module):
    def __init__(self, manifold ,f1_in=768, f2_in=768, ft_trans=[768], ft_gcn=[768, 512], ft_com=[512, 512], f1_out=256, f2_out=768,
                 type_gcn='GCN', skip=False, batch_norm=True, dropout=0.5, act_func='relu'):
        super(Poincare_MH, self).__init__()
        self.enc =  PoincareEnLiFu(
            manifold=manifold, 
            f1_in=f1_in, 
            f2_in=f2_in, 
            f1_out=f1_out, 
            f2_out=f2_out,
            ft_trans=ft_trans, 
            ft_gcn=ft_gcn, 
            ft_com=ft_com, 
            type_graph=type_gcn, 
            skip=skip, 
            dropout=dropout, 
            act_func=act_func
        )
    def forward(self, data):
        coo = to_scipy_sparse_matrix(data['edge_index'],data['edge_attr'] )
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape

        adj_matrix = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense().cuda()
        
        x = self.enc(x_1=data['ft_1'], x_2=data['ft_2'],
                     n_1=data['n_node_1'], n_2=data['n_node_2'],
                     edge_index=adj_matrix,
                     x_cls_1=data['ft_proj_1'], x_cls_2=data['ft_proj_2']) # (batch, ft_com[-1])
        return x
        

class Poincare_Discriminator(nn.Module):
    def __init__(self, manifold:PoincareBall, ft_in=512, ft_out=[128,1], dropout=0.5, batch_norm=True, act_func='relu'):
        super(Poincare_Discriminator, self).__init__()
        self.linear = []
        self.manifold = manifold
        for idx in range(len(ft_out)):
            if idx == 0:
                self.linear.append(HNNLayer(manifold, ft_in , ft_out[idx], dropout=dropout, act=get_activate_func(act_func), use_bias=True))
            elif idx == len(ft_out - 1):
                self.linear.append(HNNLayer(manifold, ft_out[idx-1] , ft_out[idx], dropout=dropout, act=get_activate_func('sigmoid'), use_bias=True))
        self.disc = nn.ModuleList(self.linear)

    def forward(self, feat1, feat2):
        dist = self.manifold.logmap0(self.manifold.sqdist(feat1, feat2))
        mul = self.manifold.logmap0(self.manifold.mobius_matvec(feat1, feat2))
        feat1 = self.manifold.logmap0(feat1)
        feat2 = self.manifold.logmap0(feat2)
        feat = torch.cat([feat1,feat2, dist, mul], dim=1) 
        self.manifold.from_euclid_to_poincare(feat)
        return self.disc(feat)


