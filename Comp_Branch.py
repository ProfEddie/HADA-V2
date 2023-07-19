import torch
import torch.nn as nn
from Comp_Basic import *
import torch.nn.functional as F
import hyptorch.nn as hypnn 

def concat_node(x1, x2, n_x1, n_x2):
    x_concat = torch.tensor(()).to(x1.device)
    count1 = 0
    count2 = 0
    print(x1)
    print(x2)
    print(n_x1)
    print(n_x2)
    for idx in range(len(n_x1)):
        x_concat = torch.cat((x_concat, x1[count1:count1+n_x1[idx]], x2[count2:count2+n_x2[idx]]), dim=0)
        count1 += n_x1[idx]
        count2 += n_x2[idx]
    return x_concat

def unconcat_node(x, n_x1, n_x2):
    n_cum = torch.cumsum(n_x1+n_x2, dim=0)
    n_x1a = torch.cat((torch.tensor([0]).to(x.device), n_cum))[:-1]
    n_x2a = n_x1a + n_x1
    x1 = x[n_x1a]
    x2 = x[n_x2a]
    return x1, x2

class LiFu(nn.Module):
    def __init__(self, f1_in=768, f2_in=768, ft_trans=[768, 768], ft_gcn=[768, 512], n_heads=4, type_graph='GCN', 
                 skip=False, batch_norm=True, dropout=0.5, act_func='relu'):
        super(LiFu, self).__init__()
        self.trans_1 = SeqLinear(f1_in, ft_out=ft_trans, 
                                 batch_norm=batch_norm, dropout=0, act_func=act_func)
        self.trans_2 = SeqLinear(f2_in, ft_out=ft_trans, 
                                 batch_norm=batch_norm, dropout=0, act_func=act_func)
        self.gcn = GraphLayer(in_channels=ft_trans[-1], hidden_channels=ft_gcn, type_model=type_graph, n_heads=n_heads,
                              skip=skip, concat=True, batch_norm=batch_norm, dropout=dropout, act_func=act_func)
    def forward(self,x_1, x_2, n_1, n_2, edge_index, edge_attr, batch_index):
        x_1 = self.trans_1(x_1) # total n_albef, ft
        x_2 = self.trans_2(x_2) # total n_dot, ft
        # concat x_albef + x_dot
        x_concat = concat_node(x_1, x_2, n_1, n_2)
        x, edge_index, edge_attr = self.gcn(x_concat, edge_index, edge_attr, batch_index)
        x_cls_1, x_cls_2 = unconcat_node(x, n_1, n_2)
        return x_cls_1, x_cls_2

class EnLiFu(nn.Module):
    def __init__(self, f1_in=768, f2_in=768, ft_trans=[768, 768], 
                 ft_gcn=[768, 512], ft_com=[512, 512], f1_out=256, f2_out=768,
                 n_heads=4, type_graph='GCN', skip=False, batch_norm=True, dropout=0.5, act_func='relu'):
        super(EnLiFu, self).__init__()
        self.gcn = LiFu(f1_in=f1_in, f2_in=f2_in, ft_trans=ft_trans, ft_gcn=ft_gcn, 
                        n_heads=n_heads, type_graph=type_graph, 
                        skip=skip, batch_norm=batch_norm, dropout=dropout, act_func=act_func)
        self.lin = SeqLinear(2*ft_gcn[-1]+f1_out+f2_out, ft_out=ft_com, batch_norm=batch_norm, 
                             dropout=dropout, act_func=act_func)
    def forward(self, x_1, x_2, n_1, n_2, edge_index, edge_attr, batch_index, x_cls_1, x_cls_2):
        g_cls_1, g_cls_2 = self.gcn(x_1, x_2, n_1, n_2, edge_index, edge_attr, batch_index)
        # x_cls = torch.cat((x_cls_albef, x_cls_dot), dim=1)
        x_enc = torch.cat((g_cls_1, x_cls_1, x_cls_2, g_cls_2),dim=1)
        x_enc = self.lin(x_enc)
        return x_enc
        
class HypEnLiFu(nn.Module):
    def __init__(self, f1_in=768, f2_in=768, ft_trans=[768, 768], 
                 ft_gcn=[768, 512], ft_com=[512, 512], f1_out=256, f2_out=768,
                 n_heads=4, type_graph='GCN', skip=False, batch_norm=True, dropout=0.5, act_func='relu', c=0.1):
        super(HypEnLiFu, self).__init__()
        self.gcn = LiFu(f1_in=f1_in, f2_in=f2_in, ft_trans=ft_trans, ft_gcn=ft_gcn, 
                        n_heads=n_heads, type_graph=type_graph, 
                        skip=skip, batch_norm=batch_norm, dropout=dropout, act_func=act_func)
        self.to_poincare = hypnn.ToPoincare(
            c=c,
            ball_dim=ft_gcn[-1],
            riemannian=False,
            clip_r=2.3,
            train_x=False,
        )
        self.lin = SeqLinear(2*ft_gcn[-1]+f1_out+f2_out, ft_out=ft_com, batch_norm=batch_norm, 
                             dropout=dropout, act_func=act_func)
    def forward(self, x_1, x_2, n_1, n_2, edge_index, edge_attr, batch_index, x_cls_1, x_cls_2):
        g_cls_1, g_cls_2 = self.gcn(x_1, x_2, n_1, n_2, edge_index, edge_attr, batch_index)
        
        # x_cls = torch.cat((x_cls_albef, x_cls_dot), dim=1)

        x_enc = torch.cat((g_cls_1, x_cls_1, x_cls_2, g_cls_2),dim=1)
        x_enc = self.lin(x_enc)
        x_enc = self.to_poincare(x_enc)
        return x_enc


class HypWithOutGraphEnLiFu(nn.Module):
    def __init__(self, f1_in=768, f2_in=768, ft_com=[512, 512], batch_norm=True, dropout=0.1, act_func='relu', c=0.1):
        super(HypWithOutGraphEnLiFu, self).__init__()
        self.to_poincare = hypnn.ToPoincare(
            c=c,
            ball_dim=f1_in+f2_in,
            riemannian=True,
            clip_r=2.3,
            train_x=False,
        )
        self.lin = HypSeqLinear(f1_in+f2_in, ft_out=ft_com, batch_norm=batch_norm, 
                             dropout=dropout, act_func=act_func)
    def forward(self, x_cls_1, x_cls_2):
        x_enc = torch.cat((x_cls_1, x_cls_2),dim=1)
        x_enc = self.to_poincare(x_enc)
        x_enc = self.lin(x_enc)
        return x_enc







# class EnLiFuEx(nn.Module):
#     def __init__(self, ft_trans=[768, 768], ft_gcn=[768, 512], ft_com=[512, 512], 
#                  n_heads=4, type_graph='GCN', skip=False, batch_norm=True, dropout=0.5, act_func='relu'):
#         super(EnLiFuEx, self).__init__()
#         self.gcn = LiFu(ft_trans=ft_trans, ft_gcn=ft_gcn, n_heads=n_heads, type_graph=type_graph, 
#                         skip=skip, batch_norm=batch_norm, dropout=dropout, act_func=act_func)
#         self.cls_a = SeqLinear(256, [256], batch_norm=batch_norm, dropout=dropout, act_func=act_func)
#         self.cls_d = SeqLinear(768, [768], batch_norm=batch_norm, dropout=dropout, act_func=act_func)
#         self.lin = SeqLinear(2*ft_gcn[-1]+1024, ft_out=ft_com, 
#                             batch_norm=batch_norm, dropout=dropout, act_func=act_func)
#     def forward(self, x_albef, x_dot, n_albef, n_dot, edge_index, edge_attr, batch_index, x_cls_albef, x_cls_dot):
#         g_cls_albef, g_cls_dot = self.gcn(x_albef, x_dot, n_albef, n_dot, edge_index, edge_attr, batch_index)
#         x_cls_albef = self.cls_a(x_cls_albef)
#         x_cls_dot = self.cls_d(x_cls_dot)
#         x_cls = torch.cat((x_cls_albef, x_cls_dot), dim=1)
#         x_enc = torch.cat((g_cls_albef, x_cls, g_cls_dot),dim=1)
#         x_enc = self.lin(x_enc)
#         return x_enc