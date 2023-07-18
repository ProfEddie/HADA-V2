from Comp_Lorentz import LorentzEnLiFu, LorentzSeqLinear 
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyptorch.lorentz.manifold import Lorentz

class Lorentz_MH(nn.Module):
    def __init__(self, manifold ,f1_in=768, f2_in=768, ft_trans=[768], ft_gcn=[768, 512], ft_com=[512, 512], f1_out=256, f2_out=768,
                 n_heads=4, type_gcn='GCN', skip=False, batch_norm=True, dropout=0.5, act_func='relu'):
        super(Lorentz_MH, self).__init__()
        self.enc =  LorentzEnLiFu(
            manifold=manifold, 
            f1_in=f1_in, 
            f2_in=f2_in, 
            f1_out=f1_out, 
            f2_out=f2_out,
            ft_trans=ft_trans, 
            ft_gcn=ft_gcn, 
            ft_com=ft_com, 
            n_heads=n_heads,
            type_graph=type_gcn, 
            skip=skip, 
            batch_norm=batch_norm, 
            dropout=dropout, 
            act_func=act_func
        )
    def forward(self, data):
        x = self.enc(x_1=data['ft_1'], x_2=data['ft_2'],
                     n_1=data['n_node_1'], n_2=data['n_node_2'],
                     edge_index=data['edge_index'], edge_attr=data['edge_attr'], 
                     x_cls_1=data['ft_proj_1'], x_cls_2=data['ft_proj_2']) # (batch, ft_com[-1])
        return x
        

class Lorentz_Discriminator(nn.Module):
    def __init__(self, manifold ,ft_in=512, ft_out=[128,1], dropout=0.5, batch_norm=True, act_func='relu'):
        super(Lorentz_Discriminator, self).__init__()
        self.disc = LorentzSeqLinear(manifold, ft_in=ft_in*4, ft_out=ft_out,
                              dropout=dropout, act_func=act_func)
    def forward(self, feat1, feat2):
        dist = torch.abs(feat1-feat2)
        mul = torch.mul(feat1, feat2)
        return torch.sigmoid(self.disc(torch.cat([feat1, feat2, dist, mul], dim=1)))


