import torch
from Utils import MemoryEfficientSwish
from hyptorch.nn import HypLinear
import torch.nn as nn
from hyptorch.lorentz.nn import LorentzGraphConvolution, HyperbolicGraphConvolution, LorentzLinear

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

def select_graph_layer(type_model='GCN'):
    if type_model == 'LGCN':
        return LorentzGraphConvolution 
    if type_model == 'LGATv2':
        return LorentzGraphConvolution 
    if type_model == 'PGCN':
        return HyperbolicGraphConvolution 

    
class LorentzSeqLinear(nn.Module):
    def __init__(self, manifold, ft_in, ft_out=[128], dropout=0.5,  act_func='relu'):
        super(LorentzSeqLinear, self).__init__()
        self.linear = []
        for idx in range(len(ft_out)):
            if idx == 0:
                self.linear.append(LorentzLinear(manifold, ft_in , ft_out[idx], 1, dropout=dropout, nonlin=None))
            else:
                self.linear.append(LorentzLinear(manifold, ft_in , ft_out[idx], 1, dropout=dropout, nonlin=get_activate_func(act_func)))
            self.dropout.append(nn.Dropout(p=dropout))
            
        self.linear = nn.ModuleList(self.linear)
        
    def forward(self, x):
        for idx in range(len(self.linear)):
            x = self.linear[idx](x)
        return x  
    




class LorentzGraphLayer(nn.Module):
    def __init__(self, manifold ,in_channels, hidden_channels=[128], type_model='GCN', n_heads=4, 
                 dropout=0.4, act_func='relu'):
        super().__init__()
        assert type_model in ['GCN', 'GATv2', 'TGCN']
        self.type_model = type_model
        self.conv = []
        self.manifold = manifold
        self.n_heads = n_heads
        # self.dropout = []
        act_func = get_activate_func(act_func=act_func)
        for idx in range(len(hidden_channels)):
            if idx == 0:
                self.conv.append(LorentzGraphConvolution(
                    manifold=manifold,
                    in_features=in_channels, 
                    out_features=hidden_channels[idx], 
                    local_agg=True,
                    use_att=True, 
                    use_bias=True,
                    dropout=dropout,
                    nonlin=act_func
                ))
                                            
            else:
               self.conv.append(LorentzGraphConvolution(
                    manifold=manifold,
                    in_features=hidden_channels[idx-1], 
                    out_features=hidden_channels[idx], 
                    local_agg=True,
                    use_att=True, 
                    use_bias=True,
                    dropout=dropout,
                    nonlin=act_func
                ))
            
        self.conv = nn.ModuleList(self.conv)
        
    def forward(self, x, edge_index, edge_attr=None):
        for idx in range(len(self.conv)):
            if edge_attr is not None:
                xout = self.conv[idx](x, edge_index, edge_attr)
            else:
                xout = self.conv[idx](x, edge_index)
            if self.skip:
                x = x + xout
            else: x = xout
            
        return x, edge_index, edge_attr # (num_nodes in a batch, hidden_channel)  


class LorentzLiFu(nn.Module):
    def __init__(self, manifold, f1_in=768, f2_in=768, ft_trans=[768, 768], ft_gcn=[768, 512], type_graph='GCN', 
                 skip=False, batch_norm=True, dropout=0.5, act_func='relu'):
        super(LorentzLiFu, self).__init__()
        self.trans_1 = LorentzSeqLinear(manifold, f1_in, ft_out=ft_trans, 
                                 batch_norm=batch_norm, dropout=0, act_func=act_func)
        self.trans_2 = LorentzSeqLinear(manifold, f2_in, ft_out=ft_trans, 
                                 batch_norm=batch_norm, dropout=0, act_func=act_func)
        self.gcn = LorentzGraphLayer(manifold, in_channels=ft_trans[-1], hidden_channels=ft_gcn, type_model=type_graph, 
                              skip=skip, batch_norm=batch_norm, dropout=dropout, act_func=act_func)
    def forward(self,x_1, x_2, n_1, n_2, edge_index, edge_attr):
        x_1 = self.trans_1(x_1) # total n_albef, ft
        x_2 = self.trans_2(x_2) # total n_dot, ft
        x_concat = concat_node(x_1, x_2, n_1, n_2)
        x, edge_index, edge_attr = self.gcn(x_concat, edge_index, edge_attr )
        x_cls_1, x_cls_2 = unconcat_node(x, n_1, n_2)
        return x_cls_1, x_cls_2


class LorentzEnLiFu(nn.Module):
    def __init__(self, manifold ,f1_in=768, f2_in=768, ft_trans=[768, 768], 
                 ft_gcn=[768, 512], ft_com=[512, 512], f1_out=256, f2_out=768,
                  type_graph='GCN', skip=False, batch_norm=True, dropout=0.5, act_func='relu'):
        super(LorentzEnLiFu, self).__init__()
        self.manifold = manifold
        self.gcn = LorentzLiFu(
            manifold, 
            f1_in=f1_in, 
            f2_in=f2_in, 
            ft_trans=ft_trans, 
            ft_gcn=ft_gcn, 
            type_graph=type_graph, 
            skip=skip, 
            batch_norm=batch_norm, 
            dropout=dropout, 
            act_func=act_func
        )
        self.lin = LorentzLinear(manifold, 2*ft_gcn[-1]+f1_out+f2_out, ft_out=ft_com,
                             dropout=dropout, act_func=act_func)
    def forward(self, x_1, x_2, n_1, n_2, edge_index, edge_attr,  x_cls_1, x_cls_2):
        x_1 = self.manifold.expmap0(x_1)
        x_2 = self.manifold.expmap0(x_2)
        n_1 = self.manifold.expmap0(n_1)
        n_2 = self.manifold.expmap0(n_2)
        x_cls_1 = self.manifold.expmap0(x_cls_1)
        x_cls_2 = self.manifold.expmap0(x_cls_2)
        g_cls_1, g_cls_2 = self.gcn(x_1, x_2, n_1, n_2, edge_index, edge_attr)
        x_enc = torch.cat((g_cls_1, x_cls_1, x_cls_2, g_cls_2),dim=1)
        x_enc = self.lin(x_enc)
        return x_enc
