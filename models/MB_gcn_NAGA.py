import math
import torch
import torch.nn as nn

from functools import reduce
import torch.nn.functional as F

'''

Ablation Study  about   2)Effect of multi-attention mechanism       2-7  NA GA attention

Three branches,  just SA NA,  no full Conection layer      
#
'''
class GlGraphConv(nn.Module):

    def __init__(self, in_features, out_features, adj, bias=True):
        super(GlGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
  
        self.W = nn.Parameter(torch.zeros(size=(3, in_features, out_features), dtype=torch.float))
        self.MM = nn.Parameter(torch.zeros(size=(16, 8), dtype=torch.float))

        nn.init.kaiming_normal_(self.W.data)
        nn.init.kaiming_normal_(self.MM.data)

        self.adj = adj #self adj just connect information
        self.m = (self.adj > 0)

        self.e = nn.Parameter(torch.zeros(1, len(self.m.nonzero()), dtype=torch.float)) 
        nn.init.constant_(self.e.data, 1)   #初始值为1        

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)
        
        self.pool= nn.MaxPool1d(kernel_size=(2))

    def forward(self, input):
        
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])
        h3 = torch.matmul(input, self.W[2])

                
        adj = -9e15 * torch.ones_like(self.adj).to(input.device)
        #adj2=adj
        adj[self.m] = self.e
        adj = F.softmax(adj, dim=1) 

        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        output = h0 + torch.matmul(adj* (1 - M), h1)   #adj2, 
  
#        
        f_div_C=F.softmax(self.MM,dim=1)#        
        h3=self.pool(h3.transpose(1, 2)).transpose(1, 2)        
        y1 = torch.matmul(f_div_C, h3)   #64 16  128        
        
        output=output+y1/2 
        

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class _GraphConv(nn.Module):
    """
    一个图卷积层，包含bn， relu, dropout
    """
    
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = GlGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x
 
    
class _ResGraphConv(nn.Module):
    """
    一个res图卷积模块，包含两个图卷积层和一个res连接
    """    
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, hid_dim, output_dim, p_dropout)
        
        #self.nonlocal_m = GraphNonLocal(hid_dim, sub_sample=1)

    def forward(self, x):
        residual = x
        
        out = self.gconv1(x)
        out = self.gconv2(out)       
        
        return residual + out
    
 


class ResGCN_model(nn.Module):
    def __init__(self, adj, hid_dim=128, coords_dim=(2, 3), num_layers=4, nodes_group=None, p_dropout=None):
        super(ResGCN_model, self).__init__()

        #graph
        group_size = len(nodes_group[0])
        assert group_size > 1

        grouped_order = list(reduce(lambda x, y: x + y, nodes_group))
        restored_order = [0] * len(grouped_order)
        for i in range(len(restored_order)):
            for j in range(len(grouped_order)):
                if grouped_order[j] == i:
                    restored_order[i] = j
                    break      
        self.grouped_order=grouped_order
        self.restored_order=restored_order
        
        #input        
        _gconv_input = [GlGraphConv(coords_dim[0], hid_dim,adj)]
        self.bn = nn.BatchNorm1d( hid_dim)
        self.relu=nn.ReLU()
        
        self.gconv_input = nn.Sequential(*_gconv_input)
        
        #mid
        _gconv_layers = []
        for i in range(num_layers):
            _gconv_layers.append(_ResGraphConv(adj, hid_dim, hid_dim, hid_dim, p_dropout=p_dropout))
            #_gconv_layers.append(TConv(hid_dim, sub_sample=2))
       
        self.gconv_layers = nn.Sequential(*_gconv_layers)
        
        #out
        self.gconv_output1 =GlGraphConv(hid_dim, coords_dim[1], adj)



    def forward(self, x):
        #print(x.size()) 
        
        b, n, c = x.size()
        #x = x.view(int(b), 16, 2)
        
        x= x[:, self.grouped_order, :]  #调整节点顺序
        #imput
        out = self.gconv_input(x).transpose(1, 2)
        out=self.bn(out).transpose(1, 2)
        out=self.relu(out)        
        #mid
        out = self.gconv_layers(out) 
        
        #out
        out=self.gconv_output1(out)       
        out = out[:, self.restored_order,:]  #恢复原来的节点顺序        
        
        return out