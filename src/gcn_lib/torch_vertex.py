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

from compressai.layers import GDN 
from positional_encodings.torch_encodings import PositionalEncodingPermute2D

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
                 loop = True,
                 aggr='mean',
                 from_tensor = True):
        
        super(PyGraph, self).__init__()
        self.k = kernel_size
        self.loop = loop
        self.conv_name = conv

        self.from_tensor = from_tensor

        if(conv == 'cheb'):
            conv_layer = nn_pyg.ChebConv(in_channels, 
                                         out_channels,
                                         K=cheb_k,
                                         normalization='sym', 
                                         bias=bias, 
                                         flow='source_to_target')
        
        elif(conv == 'gcn'):
            conv_layer = nn_pyg.GCNConv(in_channels, out_channels, flow='source_to_target')
        elif(conv == 'sage'):
            conv_layer = nn_pyg.SAGEConv(in_channels, out_channels, aggr=aggr, bias=bias, flow='source_to_target')
        else:
            raise NotImplementedError('conv:{} is not supported by PyGraph class'.format(conv))
        
        self.conv_layer = conv_layer

    def forward(self, x):

        B, C, H, W = x.shape # torch.Size([16, 3, 5, 5]
                
        x_f = flat_nodes(x, (B, C, H, W))
        batch = torch.linspace(0,B,steps=(B*H*W),dtype=torch.int64).to(device=x.device)

        edge_index = knn_graph(x_f, self.k, batch, loop=self.loop)

        
        if(self.conv_name in ['gcn', 'sage']):
            out = self.conv_layer(
                x=x_f,
                edge_index = edge_index)
        else:
            out = self.conv_layer(
                x=x_f,
                edge_index = edge_index,
                batch = batch)
            
        return out






class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, 
                 in_channels, 
                 out_channels,
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
        self.out_channels = out_channels


        self.graph_conv = PyGraph(
                                in_channels, 
                                out_channels, 
                                knn, 
                                conv, 
                                bias, 
                                cheb_k = cheb_k, 
                                loop = loop,
                                aggr = aggr)
        
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

        #self.pool = nn_pyg.TopKPooling(out_channels, ratio = 0.25)


        
    def forward(self, x):

        B, C, H, W = x.shape
        x = self.graph_conv(x)
        x = self.norm_layer(x)
        x = self.activation_layer(x)

        #x, edge_index, _, batch, _, _ = self.pool(x, edge_index, None, batch)


        return x
    

class MultiGrapher(nn.Module):
    def __init__(self, 
             in_channels, 
             out_channels,
             knn=9, 
             conv='edge',
             heads = 1, 
             aggr='mean',
             bias=True, 
             cheb_k = 2, 
             loop = True,
             recompute_graph = True):
    
        super(MultiGrapher, self).__init__()

        self.k = knn
        self.loop = loop
        self.recompute_graph = recompute_graph
        self.out_channels = out_channels

        self.p_enc_2d = PositionalEncodingPermute2D(in_channels)

        if(conv == 'transformer'):
            self.conv1 =  nn_pyg.TransformerConv(in_channels,in_channels,heads=heads,bias=bias, flow='source_to_target')
        elif(conv == 'sage'):
            self.conv1 = nn_pyg.SAGEConv(in_channels, in_channels, aggr=aggr, bias=bias, flow='source_to_target')
        elif(conv == 'cheb'):
            self.conv1 = nn_pyg.ChebConv(in_channels, 
                                         in_channels,
                                         K=cheb_k,
                                         normalization='sym', 
                                         bias=bias, 
                                         flow='source_to_target')
        else:
            raise NotImplementedError('conv:{} is not supported by MultiGrapher class'.format(conv))

        self.pool1 = nn_pyg.TopKPooling(in_channels, ratio = 0.25)

        self.norm = GDN(in_channels)

        if(conv == 'transformer'):
            self.conv2 =  nn_pyg.TransformerConv(in_channels,out_channels,heads=heads,bias=bias, flow='source_to_target')
        elif(conv == 'sage'):
            self.conv2 = nn_pyg.SAGEConv(in_channels, out_channels, aggr=aggr, bias=bias, flow='source_to_target')
        elif(conv == 'cheb'):
            self.conv2 = nn_pyg.ChebConv(in_channels, 
                                         out_channels,
                                         K=cheb_k,
                                         normalization='sym', 
                                         bias=bias, 
                                         flow='source_to_target')
        else:
            raise NotImplementedError('conv:{} is not supported by MultiGrapher class'.format(conv))
            
        self.pool2 = nn_pyg.TopKPooling(out_channels, ratio = 0.25)

    def forward(self, x):

        B, C, H, W = x.shape # torch.Size([16, 3, 5, 5]

        pos_enc = self.p_enc_2d(x)

        x = pos_enc + x
        
        x = flat_nodes(x, (B, C, H, W))
        batch = torch.linspace(0,B,steps=(B*H*W),dtype=torch.int64).to(device=x.device)

        edge_index = knn_graph(x, self.k, batch, loop=self.loop)

        x = self.conv1(x, edge_index)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)


        x = unflat_nodes(x, (B,C,H//2,W//2))
        x = self.norm(x)
        x = flat_nodes(x, (B, C, H//2, W//2))
        
        if(self.recompute_graph):
            edge_index = knn_graph(x, self.k, batch, loop=self.loop)

        x = self.conv2(x, edge_index)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)

        x = unflat_nodes(x, (B,self.out_channels,H//4,W//4))

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
            #nn.BatchNorm2d(out_dim),
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
    



# IGNORE
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
    



if __name__ == '__main__':
    device = "cuda" if  torch.cuda.is_available() else "cpu"
    x = torch.rand((8,128,64,64))
    x= x.to(device)

    net = MultiGrapher(
        in_channels=128,
        out_channels=192,
        knn=9,
        conv='sage',
        aggr='max',
        bias=True,
        loop=True,
        recompute_graph=False
    ).to(device)

    out = net(x)
    print(out.shape)