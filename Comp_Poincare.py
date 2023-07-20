import torch
from Utils import MemoryEfficientSwish
from hyptorch.nn import HypLinear, ConcatPoincareLayer
import torch.nn as nn
from hyptorch.nn import HNNLayer, HypAct, HypAgg, HyperbolicGraphConvolution, HypLinear 
from hyptorch.poincare.manifold import PoincareBall
from Comp_Basic import *

def concat_node(x1, x2, n_x1, n_x2):
    x_concat = torch.tensor(()).to(x1.device)
    count1 = 0
    count2 = 0
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

def get_activate_func(act_func=None):
    if act_func is None or act_func.lower() == 'id':
        return nn.Identity()
    if act_func.lower() == 'relu':
        return nn.ReLU()
    if act_func.lower() == 'swish':
        return MemoryEfficientSwish()
    if act_func.lower() == 'tanh':
        return nn.Tanh()
    if act_func.lower() == 'gelu':
        return nn.GELU()
    if act_func.lower() == 'elu':
        return nn.ELU()
    if act_func.lower() == 'sigmoid':
        return nn.Sigmoid()

def select_graph_layer(type_model='GCN'):
        return HyperbolicGraphConvolution 

    
class PoincareSeqLinear(nn.Module):
    def __init__(self, manifold, ft_in, ft_out=[128], dropout=0.5,  act_func='relu'):
        super(PoincareSeqLinear, self).__init__()
        self.linear = []
        for idx in range(len(ft_out)):
            if idx == 0:
                self.linear.append(HNNLayer(manifold, ft_in , ft_out[idx], dropout=dropout, act=get_activate_func(act_func), use_bias=True))
            else:
                self.linear.append(HNNLayer(manifold, ft_out[idx-1] , ft_out[idx], dropout=dropout, act=get_activate_func(act_func), use_bias=True))
            
        self.linear = nn.ModuleList(self.linear)
        
    def forward(self, x):
        for idx in range(len(self.linear)):
            x = self.linear[idx](x)
        return x  
    
 

class PoincareGraphLayer(nn.Module):
    def __init__(self, manifold ,in_channels, hidden_channels=[128], type_model='GCN', skip=False, dropout=0.4, act_func='relu'):
        super().__init__()
        assert type_model in ['GCN', 'GATv2', 'TGCN']
        self.type_model = type_model
        self.conv = []
        self.manifold = manifold
        self.skip = skip
        act_func = get_activate_func(act_func=act_func)
        for idx in range(len(hidden_channels)):
            if idx == 0:
                self.conv.append(HyperbolicGraphConvolution(
                    manifold=manifold,
                    in_features=in_channels, 
                    out_features=hidden_channels[idx], 
                    local_agg=False,
                    use_att=False, 
                    use_bias=True,
                    dropout=dropout,
                    act=act_func
                ))
                                            
            else:
               self.conv.append(HyperbolicGraphConvolution(
                    manifold=manifold,
                    in_features=hidden_channels[idx-1], 
                    out_features=hidden_channels[idx], 
                    local_agg=False,
                    use_att=False, 
                    use_bias=True,
                    dropout=dropout,
                    act=act_func
                ))
            
        self.conv = nn.ModuleList(self.conv)
        
    def forward(self, x, edge_index):
        for idx in range(len(self.conv)):
            xout, adj = self.conv[idx](x, edge_index)
            if self.skip:
                x = x + xout
            else: x = xout
            
        return x, adj # (num_nodes in a batch, hidden_channel)  


class PoincareLiFu(nn.Module):
    def __init__(self, manifold:PoincareBall, f1_in=768, f2_in=768, ft_trans=[768, 768], ft_gcn=[768, 512], type_graph='GCN', 
                 skip=False, dropout=0.5, act_func='relu'):
        super(PoincareLiFu, self).__init__()
        self.manifold = manifold
        self.trans_1 = SeqLinear(f1_in, ft_out=ft_trans, 
                                 batch_norm=True, dropout=0, act_func=act_func)
        self.trans_2 = SeqLinear(f2_in, ft_out=ft_trans, 
                                 batch_norm=True, dropout=0, act_func=act_func)
        self.gcn = PoincareGraphLayer(manifold, in_channels=ft_trans[-1], hidden_channels=ft_gcn, type_model=type_graph, 
                              skip=skip, dropout=dropout, act_func=act_func)
    def forward(self,x_1, x_2, n_1, n_2, edge_index ):
        x_1 = self.trans_1(x_1) # total n_albef, ft
        x_2 = self.trans_2(x_2) # total n_dot, ft

        x_concat = self.manifold.from_euclid_to_poincare(concat_node(x_1, x_2, n_1, n_2))
        x, edge_index = self.gcn(x_concat, edge_index)
        x_cls_1, x_cls_2 = unconcat_node(self.manifold.logmap0(x), n_1, n_2)
        x_cls_1 = self.manifold.from_euclid_to_poincare(x_cls_1)
        x_cls_2 = self.manifold.from_euclid_to_poincare(x_cls_2)
        return x_cls_1, x_cls_2


class PoincareEnLiFu(nn.Module):
    def __init__(self, manifold:PoincareBall ,f1_in=768, f2_in=768, ft_trans=[768, 768], 
                 ft_gcn=[768, 512], ft_com=[512, 512], f1_out=256, f2_out=768,
                  type_graph='GCN', skip=False, dropout=0.5, act_func='relu'):
        super(PoincareEnLiFu, self).__init__()
        self.manifold = manifold
        self.gcn = PoincareLiFu(
            manifold, 
            f1_in=f1_in, 
            f2_in=f2_in, 
            ft_trans=ft_trans, 
            ft_gcn=ft_gcn, 
            type_graph=type_graph, 
            skip=skip, 
            dropout=dropout, 
            act_func=act_func
        )
        self.lin = PoincareSeqLinear(
            manifold, 
            ft_in=2*ft_gcn[-1]+f1_out+f2_out, 
            ft_out=ft_com,
            dropout=dropout, 
            act_func=act_func
        )
        self.cat_layer_1 = ConcatPoincareLayer(manifold, f1_out, ft_gcn[-1], ft_gcn[-1]+f1_out) 
        self.cat_layer_2 = ConcatPoincareLayer(manifold, f2_out, ft_gcn[-1], ft_gcn[-1]+f2_out) 
        self.cat_layer_all = ConcatPoincareLayer(manifold, f1_out +ft_gcn[-1], f2_out + ft_gcn[-1], 2*ft_gcn[-1]+f1_out+f2_out) 
    def forward(self, x_1, x_2, n_1, n_2, edge_index, x_cls_1, x_cls_2):
        g_cls_1, g_cls_2 = self.gcn(x_1, x_2, n_1, n_2, edge_index )
        x_cls_1 = self.manifold.from_euclid_to_poincare(x_cls_1) 
        x_cls_2 = self.manifold.from_euclid_to_poincare(x_cls_2) 
        x_g_cls_1 = self.cat_layer_1(x_cls_1, g_cls_1)
        x_g_cls_2 = self.cat_layer_2(x_cls_2, g_cls_2)
        x_enc = self.cat_layer_all(x_g_cls_1, x_g_cls_2)
        x_enc = self.lin(x_enc)
        return x_enc
