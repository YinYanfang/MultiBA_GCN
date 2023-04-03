import math
import torch
import torch.nn as nn

from functools import reduce
import torch.nn.functional as F


'''

Ablation Study  about   3)Effect of full connection layer    3-4

two branches + Full connect layer  

'''

class GlGraphConv(nn.Module):
    """
    #sem_Graph, 思想是 st-gcn,  借鉴了 st-gcn，并且已经包含了 ro i 信息，就是学习的边信息
    全局信息采用了下采样, 全局信息 phi thei 是特征提取的
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(GlGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        '''已经借鉴了st-gcn中，节点自身 和 相邻节点 之间，具有不同的学习参数 size=2两个W矩阵 '''       
        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))

        nn.init.kaiming_normal_(self.W.data)


        self.adj = adj #self adj just connect information
        self.m = (self.adj > 0)


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
  
       
        adj =self.adj.to(input.device)

        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1)        
        
  
        

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

        self.gconv_output1 =GlGraphConv(hid_dim, int(hid_dim//2), adj)       
        self.gconv_output2=nn.Linear(16*int(hid_dim//2),16*coords_dim[1])   
        


    def forward(self, x):
        #print(x.size()) 
        
        b, n, c = x.size()

        
        x= x[:, self.grouped_order, :]  #调整节点顺序
        #imput
        out = self.gconv_input(x).transpose(1, 2)
        out=self.bn(out).transpose(1, 2)
        out=self.relu(out)        
        #mid
        out = self.gconv_layers(out) 
        
        #out
        out=self.gconv_output1(out)
       
        # fcl
        b,n,c=out.size()
        out=out.view(b,c*n)
        out=self.gconv_output2(out)         
        out=out.view(b,n,3)      
        
        #out = out[:, self.restored_order,:]  #恢复原来的节点顺序 
        

        
        return out