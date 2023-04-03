import math
import torch
import torch.nn as nn

from functools import reduce
import torch.nn.functional as F


'''
Final model: multi branches, 3 Attentions(SA,NA,GA), full conection,  sum fusion

Comparison with GCN method  4.3.1-7     

Ablation Study about  
                      3) Effect of full connection layer FCL             3-6
                      4)  Ablation experiments of the network depths     :all use this model, just layer is different
                      5) Ablation experiments of feature fusion methods  : sum fusion


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
        self.W = nn.Parameter(torch.zeros(size=(3, in_features, out_features), dtype=torch.float))
        self.MM = nn.Parameter(torch.zeros(size=(16, 8), dtype=torch.float))

        nn.init.kaiming_normal_(self.W.data)
        nn.init.kaiming_normal_(self.MM.data)

        self.adj = adj #self adj just connect information
        self.m = (self.adj > 0)
        '''#图的边,e,是一个可学习的参数'''
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
        
        ''' 归一化邻接矩阵, 邻接矩阵根据学习到的边的情况，是不断变化的'''
                
        adj = -9e15 * torch.ones_like(self.adj).to(input.device)
        adj[self.m] = self.e
        adj = F.softmax(adj, dim=1) 

        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1)        
        
  
        
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
        
        
        #nolocal=self.nonlocal_m(x.transpose(1, 2)).transpose(1, 2)
        
        return residual + out
    
class T2Conv(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=1, sub_sample=False, bn_layer=True):
        super(TConv, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g1 =nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0) 
        self.g2=nn.Conv1d(in_channels=self.inter_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0)       
        
        self.bn=nn.BatchNorm1d(self.in_channels)

        #self.M = nn.Parameter(torch.ones(1, 16), dtype=torch.float)  #固定的16个节点
        '''卷积参数初始化'''
        self.MM = nn.Parameter(torch.zeros(size=(16, 8), dtype=torch.float))
        nn.init.kaiming_normal_(self.MM.data)   
        nn.init.kaiming_normal_(self.g1.weight)
        nn.init.kaiming_normal_(self.g2.weight)

    def forward(self, x, return_nl_map=False):     


        y = self.g1(x)       
        y = self.g2(y)
        y=self.bn(y)        
        z = y + x
        return z

class TConv(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=1, sub_sample=False, bn_layer=True):
        super(TConv, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        

        self.g1 =nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        
            
        self.g2 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        
        self.g3=nn.Conv1d(in_channels=self.inter_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0)
        
        
        self.bn=nn.BatchNorm1d(self.in_channels)
        self.max_pool = nn.MaxPool1d(kernel_size=2)
        #self.M = nn.Parameter(torch.ones(1, 16), dtype=torch.float)  #固定的16个节点
        '''卷积参数初始化'''
        self.MM = nn.Parameter(torch.zeros(size=(16, 8), dtype=torch.float))
        nn.init.kaiming_normal_(self.MM.data)   
        nn.init.kaiming_normal_(self.g1.weight)
        nn.init.kaiming_normal_(self.g2.weight)
        nn.init.kaiming_normal_(self.g3.weight)  

    def forward(self, x, return_nl_map=False):     

        #batch_size = x.size(0)

        g_x1 = self.g1(x)
        g_x1 = self.max_pool(g_x1)        
        g_x1 = g_x1.permute(0, 2, 1)
        
        f_div_C=F.softmax(self.MM,dim=1)    
        
        y1 = torch.matmul(f_div_C, g_x1)   #64 16  64
        y1 = y1.permute(0, 2, 1).contiguous()  #64 64  16        
        
        y2 = self.g2(x)
               
        y=y1+y2  
        
        #y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        y3 = self.g3(y)
        y3=self.bn(y3)
        
        z = y3 + x

        return z

    
class _GraphNonLocal(nn.Module):
    """
    一个非局部运算模块，可以理解为全局注意力
    """      
    def __init__(self, hid_dim,sub_sample=1):
        super(_GraphNonLocal, self).__init__()
        
        self.nonlocal_m = TConv(hid_dim, sub_sample=sub_sample)

    def forward(self, x):
       
        out = self.nonlocal_m(x.transpose(1, 2)).transpose(1, 2)
       
        return out 

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
            _gconv_layers.append(TConv(hid_dim, sub_sample=2))
       
        self.gconv_layers = nn.Sequential(*_gconv_layers)
        
        #out
        self.gconv_output1 =GlGraphConv(hid_dim, int(hid_dim//2), adj)

        #self.gconv_output1 =_GraphConv(adj, hid_dim, int(hid_dim//2), p_dropout=p_dropout)
        self.gconv_output2=nn.Linear(16*int(hid_dim//2),16*coords_dim[1])   
        #self.gconv_output = SemGraphConv5(hid_dim, coords_dim[1],adj)


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
        #print(out.size())
        b,n,c=out.size()
        out=out.view(b,c*n)
        out=self.gconv_output2(out)         
        out=out.view(b,n,3)      
        
        #out = out[:, self.restored_order,:]  #恢复原来的节点顺序 
        

        
        return out