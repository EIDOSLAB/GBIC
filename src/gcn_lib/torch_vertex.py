# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import numpy as np
import torch
from torch import nn
from .torch_nn import BasicConv, batched_index_select, act_layer
from .torch_edge import DenseDilatedKnnGraph
from .pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F

from torch_geometric.typing import OptPairTensor
from .pyg_utils import *
from torch_geometric.nn.conv import SAGEConv, ChebConv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn import knn, knn_graph

import torch_geometric.nn as nn_pyg

import sys
from torch import Tensor


class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act=None, norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        # x: torch.Size([16, 4, 25, 1])
        # edge_index: torch.Size([2, 16, 25, 9])
        x_i = batched_index_select(x, edge_index[1]) # torch.Size([16, 4, 25, 9]) ->    x_i[0,:,:,0] == x_i[0,:,:,1] 
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0]) # torch.Size([16, 4, 25, 9]) having self-loops
        
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True) # torch.Size([16, 4, 25, 1])

        b, c, n, _ = x.shape
        # x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _) # torch.Size([16, 8, 25, 1])
        x = torch.cat([x,x_j], dim=1)
        return self.nn(x)


class MRConvAtt(nn.Module):
    def __init__(self, in_channels, out_channels, act=None, norm=None, bias=True, heads = 1):
        super(MRConvAtt, self).__init__()
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias)
        self.att = nn.Parameter(torch.Tensor(1, in_channels, heads, 1, 1))
        self.soft_act = nn.Softmax(dim=-1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.att)

    def forward(self, x, edge_index, y=None):
        # x: torch.Size([16, 48, 25, 1])
        # edge_index: torch.Size([2, 16, 25, 9])
        x_i = batched_index_select(x, edge_index[1]) # torch.Size([16, 48, 25, 9])

        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])

        x_diff = x_j-x_i # torch.Size([16, 48*2, 25, 9])
        x_diff = F.leaky_relu(x_diff,0.2)

        alpha = (x_diff.unsqueeze(2) * self.att).sum(dim=1) #torch.Size([16, 3, 25, 9])
        alpha = self.soft_act(alpha)

        x_max,_ = torch.max(((x_j-x_i).unsqueeze(dim=1)*alpha.unsqueeze(dim=2)).mean(dim=1),dim=-1,keepdim=True) # torch.Size([16, 48, 25, 1])

        return self.nn(torch.cat([x,x_max],dim=1))
    



class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        # x: torch.Size([16, 3, 25, 1])
        # edge_index: torch.Size([2, 16, 25, 9])
        x_i = batched_index_select(x, edge_index[1]) # torch.Size([16, 3, 25, 9])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class GraphSAGE(nn.Module):
    """
    GraphSAGE Graph Convolution (Paper: https://arxiv.org/abs/1706.02216) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GraphSAGE, self).__init__()
        self.nn1 = BasicConv([in_channels, in_channels], act, norm, bias)
        self.nn2 = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        # x: torch.Size([16, 3, 25, 1])
        # edge_index: torch.Size([2, 16, 25, 9])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0]) # torch.Size([16, 3, 25, 9])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(self.nn1(x_j), -1, keepdim=True)
        return self.nn2(torch.cat([x, x_j], dim=1)) # torch.Size([16, 8, 25, 1])


class GINConv2d(nn.Module):
    """
    GIN Graph Convolution (Paper: https://arxiv.org/abs/1810.00826) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(GINConv2d, self).__init__()
        self.nn = BasicConv([in_channels, out_channels], act, norm, bias)
        eps_init = 0.0
        self.eps = nn.Parameter(torch.Tensor([eps_init]))

    def forward(self, x, edge_index, y=None):
        # x: torch.Size([16, 3, 25, 1])
        # edge_index: torch.Size([2, 16, 25, 9])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0]) # torch.Size([16, 3, 25, 9])

        x_j = torch.sum(x_j, -1, keepdim=True) # torch.Size([16, 3, 25, 1])
        return self.nn((1 + self.eps) * x + x_j)


class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge',heads=1, act=None, norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'sage':
            self.gconv = GraphSAGE(in_channels, out_channels, act, norm, bias)
        elif conv == 'gin':
            self.gconv = GINConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mrgat':
            self.gconv = MRConvAtt(in_channels, out_channels, act, norm, bias, heads=heads)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index, y=None):
        # x: torch.Size([16, 3, 25, 1])
        # edge_index: torch.Size([2, 16, 25, 9])
        return self.gconv(x, edge_index, y)


class DyGraphConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge',heads =1, act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, conv,heads, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        self.r = r # reduce ratio 1 for vig
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x, relative_pos=None):
        B, C, H, W = x.shape # torch.Size([16, 3, 5, 5])
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()            
        x = x.reshape(B, C, -1, 1).contiguous() # torch.Size([16, 3, 25, 1])
        edge_index = self.dilated_knn_graph(x, y, relative_pos) # torch.Size([2, 16, 25, 9])
        x = super(DyGraphConv2d, self).forward(x, edge_index, y)
      
        return x.reshape(B, -1, H, W).contiguous()
    



class PyGraphBipartite(nn.Module):
    """
    Pytorch Geometric - KNN w/ pytorch geometric utils for bi-partite graph
    """
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size=9,
                conv='sage',
                heads =1,
                bias=True,
                r=1,
                loop = True,
                aggr = 'mean',
                slope = 0.01):
        super(PyGraphBipartite, self).__init__()
        self.k = kernel_size
        self.loop = loop
        self.r = r 
        self.conv_name = conv

        if(conv == 'gat'):
            conv_layer =nn_pyg.GATv2Conv((in_channels,in_channels),out_channels,heads=heads,bias=bias, flow='source_to_target')
        elif(conv == 'sage'):
            conv_layer = nn_pyg.SAGEConv((in_channels,in_channels),out_channels, aggr=aggr, bias=bias, flow='source_to_target')
        elif(conv == 'gconv'):
            conv_layer = nn_pyg.GraphConv((in_channels,in_channels),out_channels, aggr=aggr, bias=bias, flow='source_to_target')
        elif(conv == 'edge'):
            conv_layer = nn_pyg.EdgeConv(
                nn=nn.Sequential(
                    nn.Linear(2*in_channels, out_channels),
                    nn.LeakyReLU(slope),
                    nn.Linear(out_channels,out_channels)
                ),
                aggr=aggr,
                flow='source_to_target'
            )
        elif(conv == 'transformer'):
            conv_layer =  nn_pyg.TransformerConv((in_channels,in_channels),out_channels,heads=heads,bias=bias, flow='source_to_target')
                
        else:
            raise NotImplementedError('conv:{} is not supported by PyGraphBipartite class'.format(conv))
        
        self.conv_layer = conv_layer

        self.linear_heads = nn.Linear(out_channels*heads, out_channels)
        

    def forward(self, x):
        #print('\nStarting Grapher')
        #print(x.shape)
        #print(self.r)
        B, C, H, W = x.shape # torch.Size([16, 3, 5, 5])
        Hy, Wy = H, W
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            _, _, Hy, Wy = y.shape
            #print(y.shape)

            y_f = flat_nodes(y, (B, C, Hy, Wy))
            batches_y = torch.linspace(0,B,steps=(B*Hy*Wy),dtype=torch.int64).to(device=y.device)
            
        x_f = flat_nodes(x, (B, C, H, W))
        batches_x = torch.linspace(0,B,steps=(B*H*W),dtype=torch.int64).to(device=x.device)

        if self.r > 1:
            assign_index = knn(y_f, x_f, self.k, batches_y, batches_x)
            assign_index = torch.flip(assign_index, [0,1])
            nodes : OptPairTensor = (y_f,x_f)
        else:
            assign_index = knn_graph(x_f, self.k, batches_x, loop=self.loop)
            nodes : OptPairTensor = (x_f,x_f)
        
        #print(x_f.shape)
        #if(self.r > 1):
            #print(y_f.shape)
        #print(assign_index.t()[:9])
        #print(torch.max(assign_index[0]))
        #print(torch.min(assign_index[0]))
        #print()
        #print(torch.max(assign_index[1]))
        #print(torch.min(assign_index[1]))
        out = self.conv_layer(nodes, assign_index)
        #print(out.shape)

        #print('Ending grapher \n')

        if(self.conv_name in ['gat', 'transformer']):
            out = self.linear_heads(out)
        
        return out

class PyGraph(nn.Module):
    """
    Pytorch Geometric - KNN w/ pytorch geometric utils
    """
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=9, 
                 conv='cheb',
                 bias=True, 
                 cheb_k = 2, 
                 loop = True):
        
        super(PyGraph, self).__init__()
        self.k = kernel_size
        self.loop = loop
        self.conv_name = conv

        if(conv == 'cheb'):
            conv_layer = nn_pyg.ChebConv(in_channels, 
                                         out_channels,
                                         K=cheb_k,
                                         normalization='sym', 
                                         bias=bias, 
                                         flow='source_to_target')
        
        elif(conv == 'gcn'):
            conv_layer = nn_pyg.GCNConv(in_channels, out_channels, flow='source_to_target')
        else:
            raise NotImplementedError('conv:{} is not supported by PyGraph class'.format(conv))
        
        self.conv_layer = conv_layer

    def forward(self, x):
        #print('\nStarting Grapher')
        #print(x.shape)
        #print(self.r)
        B, C, H, W = x.shape # torch.Size([16, 3, 5, 5]
            
        x_f = flat_nodes(x, (B, C, H, W))
        batches_x = torch.linspace(0,B,steps=(B*H*W),dtype=torch.int64).to(device=x.device)

        assign_index = knn_graph(x_f, self.k, batches_x, loop=self.loop)

        
        #print(x_f.shape)
        #if(self.r > 1):
            #print(y_f.shape)
        #print(assign_index.t()[:9])
        #print(torch.max(assign_index[0]))
        #print(torch.min(assign_index[0]))
        #print()
        #print(torch.max(assign_index[1]))
        #print(torch.min(assign_index[1]))

        if(self.conv_name == 'gcn'):
            out = self.conv_layer(
                x=x_f,
                edge_index = assign_index)
        else:
            out = self.conv_layer(
                x=x_f,
                edge_index = assign_index,
                batch = batches_x)
        #print(out.shape)

        #print('Ending grapher \n')
        
        return out


class TorchGraph(nn.Module):
    """
    Pytorch Geometric - KNN w/ pointcloud utils
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, conv='cheb',heads =1,
                 norm='sym', bias=True, cheb_k = 2, loop = True):
        super(TorchGraph, self).__init__()
        self.in_channels = in_channels
        self.k = kernel_size
        self.loop = loop
        self.dilated_knn_graph = DenseDilatedKnnGraph(self.k)
        if(conv == 'cheb'):
            self.conv = ChebConv(in_channels, out_channels,K=cheb_k,normalization=norm, bias=bias, flow='source_to_target')
        else:
            raise NotImplementedError('conv:{} is not supported by TorchGraph class'.format(conv))

    def forward(self, x):
        B, C, H, W = x.shape
        x_f = x.reshape(B,C,-1,1)
        edge_index = self.dilated_knn_graph(x_f,None,None)

        x_j = edge_index[0]
        x_i = edge_index[1]

    

        count_batches = torch.linspace(0,B,steps=(9*H*W*B),dtype=torch.int64).to(device=x.device)

        xx_j = x_j.reshape(-1) + ( count_batches  * (H*W))
        xx_i = x_i.reshape(-1) + ( count_batches  * (H*W))
        new_edge_index = torch.cat([xx_j.unsqueeze(0),xx_i.unsqueeze(0)], dim = 0)

        print(torch.max(new_edge_index[0]), torch.max(new_edge_index[1]))
        print(torch.min(new_edge_index[0]), torch.min(new_edge_index[1]))
        print()

        x_f = flat_nodes(x, x.shape)


        out = self.conv(
            x=x_f,
            edge_index = new_edge_index)
        
        return out




class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, 
                 in_channels, 
                 knn=9, 
                 bipartite = True, 
                 conv='edge',
                 heads = 1, 
                 act='relu', 
                 slope = 0.2,
                 aggr='mean',
                 norm='none',
                 bias=True, 
                 r=1, 
                 cheb_k = 2, 
                 loop = True,
                 use_fc = False):
        
        super(Grapher, self).__init__()
        self.channels = in_channels # node's features
        self.norm = norm
        self.r = r 
        self.use_fc = use_fc

        out_channels = in_channels
        if(use_fc):
            out_channels = in_channels * 2


        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels)
        ) 
        
        if(bipartite):
            self.graph_conv = PyGraphBipartite(
                                in_channels,
                                out_channels,
                                knn,
                                conv,
                                heads,
                                bias,
                                r,
                                loop=loop,
                                aggr=aggr)
        else:
            self.graph_conv = PyGraph(
                                in_channels, 
                                out_channels, 
                                knn, 
                                conv, 
                                bias, 
                                cheb_k = cheb_k, 
                                loop = loop)
        
        # graph conv norm layer
        norm_layer = None
        if(norm == 'batch'):
            norm_layer = nn_pyg.BatchNorm(out_channels)
        elif(norm == 'instance'):
            norm_layer = nn_pyg.InstanceNorm(out_channels)
        elif(norm == 'layer'):
            norm_layer = nn_pyg.LayerNorm(out_channels)
        elif(norm == 'graph'):
            norm_layer = nn_pyg.GraphNorm(out_channels)
        else:
            norm_layer = nn.Identity()

        self.norm_layer = norm_layer

        self.activation_layer = None

        if(act == 'relu'):
            self.activation_layer = nn.ReLU()
        elif(act == 'leakyrelu'):
            self.activation_layer = nn.LeakyReLU(slope)
        elif(act == 'gelu'):
            self.activation_layer = nn.GELU()
        else:
            self.activation_layer = nn.Identity()

        

        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels)
        ) 

        
    def forward(self, x):
        if(self.use_fc):
            x = self.fc1(x)

        B, C, H, W = x.shape
        x = self.graph_conv(x)
        x = self.norm_layer(x)
        x = self.activation_layer(x)

        if(self.use_fc):
            C = C*2

        x = unflat_nodes(x, (B,C,H,W))

        #print(x.shape)
        #print('end Gnn layer')

        if(self.use_fc):
            x = self.fc2(x)
        return x
            

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )

        self.act = nn.Identity()
        if(act != 'none'):
            self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Downsample(nn.Module):
    """ Convolution-based downsample
    """
    def __init__(self, in_dim=48, out_dim=96):
        super().__init__()        
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )


    def forward(self, x):
        x = self.conv(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_dim = 96, out_dim = 48):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_dim,
                out_dim,
                kernel_size=3,
                stride=2,
                output_padding=2 - 1,
                padding=1,
            ),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.deconv(x)
        return x
    

