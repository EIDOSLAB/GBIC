# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import numpy as np
import torch
from torch import nn
from .torch_nn import BasicConv, batched_index_select, act_layer
from .torch_edge import DenseDilatedKnnGraph
from .pos_embed import get_2d_relative_pos_embed
import torch.nn.functional as F


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


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """
    def __init__(self, in_channels, knn=9, dilation=1, conv='edge',heads = 1, act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196, relative_pos=False):
        super(Grapher, self).__init__()
        self.channels = in_channels # node's features
        self.n = n # number of nodes
        self.r = r # reduce ratio: [4, 2, 1, 1] ; vig: 1 (does not reduce)
        """ self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        ) """
        self.graph_conv = DyGraphConv2d(in_channels, in_channels, knn, dilation, conv,heads,
                              act, norm, bias, stochastic, epsilon, r)
        """ self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        ) """
        self.relative_pos = None
        if relative_pos and False: # GS edit -> avoid relative_pos usage
            print('using relative_pos')
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                int(n**0.5)))).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, H, W):
        return None # GS edit -> avoid relative_pos usage
        if relative_pos is None or H * W == self.n:
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x):
        # x = self.fc1(x)
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)
        x = self.graph_conv(x,relative_pos)
        # x = self.fc2(x)
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
        #self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )

    def forward(self, x):
        x = self.fc1(x)
        #x = self.act(x)
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
            )
        )

    def forward(self, x):
        x = self.deconv(x)
        return x