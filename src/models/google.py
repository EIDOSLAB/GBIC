import warnings
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

#from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck   #, GaussianConditional
from compressai.layers import GDN  #, MaskedConv2d
from compressai.registry import register_model

from compressai.models.base import (
    CompressionModel,
 
)
from utils.functions import conv, deconv, conv_graph

from compressai.zoo import (
    bmshj2018_factorized,

)



__all__ = [
    "CompressionModel",
    "FactorizedPrior",
    "FactorizedPriorReLU"
]



@register_model("graphbased-factorized")
class GBIC_FactorizedPrior(CompressionModel):
    r"""ADD GRAPH MODULES
    """

    def __init__(self, 
                 N, 
                 M,
                 n_graph_encoder = 2,
                 symmetric = False,
                 
                 conv_type='sage',
                 bipartite = True,
                 cheb_k = 2,
                 heads = 1,
                 activation = 'none',
                 aggr = 'mean',
                 knn = 9,
                 loop = True,
                 use_ffn =False,
                 use_fc = False,
                 graph_norm = 'none',
                 **kwargs):
        
        super().__init__(**kwargs)

        self.entropy_bottleneck = EntropyBottleneck(M)


        self.g_a = nn.Sequential(
            conv(3, N, use_graph=False),
            GDN(N),

            conv(N, N, use_graph=False),
            GDN(N),

            conv_graph(
                 N, 
                 M,
                 conv=conv_type,
                 cheb_k=cheb_k, 
                 heads=heads,
                 activation=activation,
                 aggr=aggr,
                 k=knn,
                 loop=loop,
                 ratio=[4,1], 
                 norm=graph_norm)
        )

        self.g_s = nn.Sequential(
            deconv(M, 
                   N,
                   use_graph=False),
            GDN(N, inverse=True),

            deconv(N, 
                   N,
                   use_graph=False),
            GDN(N, inverse=True),

            deconv(N, 
                   N,
                   use_graph=False),
            GDN(N, inverse=True),

            deconv(N, 3)
        )

        self.N = N
        self.M = M

    @property
    def downsampling_factor(self) -> int:
        return 2**4

    def forward(self, x):
        y = self.g_a(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}





@register_model("GBIC-factorized-relu")
class GBIC_FactorizedPriorReLU(GBIC_FactorizedPrior):
    r"""add module
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, 3),
        )


if __name__ == '__main__':


    image_models = {
        "graph-factorized":GBIC_FactorizedPrior,
        "bmshj2018-factorized": bmshj2018_factorized,
    }   
    device = "cuda" if  torch.cuda.is_available() else "cpu"
    x = torch.rand((8,3,256,256))
    x= x.to(device)

    for N in [128]:
        for n_graph_encoder in [3]:
            for symmetric in [True,False]:
                #'cheb_2','cheb_3','cheb_4','gcn','gat_1','gat_2',
                for conv_layer_full in ['sage','gconv','edge','transformer_1','transformer_2','transformer_4']:
                    for activation in ['none']:#,'relu','leakyrelu','gelu']:
                        for aggr in ['mean']:#,'max']:
                            for knn in [18]:
                                for loop in [True]:#,False]:
                                    for use_ffn in [True]:#,False]:
                                        for use_fc in [True]:#,False]:
                                            for graph_norm in ['graph']:#['none','batch','instance','layer','graph']:

                                                print(n_graph_encoder,
                                                symmetric,
                                                conv_layer_full,
                                                activation,
                                                aggr,
                                                knn,
                                                loop,
                                                use_ffn,
                                                use_fc,
                                                graph_norm)
                                                
                                                heads =1
                                                cheb_k=1

                                                if(conv_layer_full.startswith('gat')):
                                                    conv_layer = conv_layer_full.split('_')[0]
                                                    heads = int(conv_layer_full.split('_')[1])
                                                elif(conv_layer_full.startswith('transformer')):
                                                    conv_layer = conv_layer_full.split('_')[0]
                                                    heads = int(conv_layer_full.split('_')[1])
                                                elif(conv_layer_full.startswith('cheb')):
                                                    conv_layer = conv_layer_full.split('_')[0]
                                                    cheb_k = int(conv_layer_full.split('_')[1])
                                                else:
                                                    conv_layer = conv_layer_full
                                                
                                                bipartite = True
                                                if(conv_layer in ['cheb', 'gcn']):
                                                    bipartite = False

                                                net = image_models['graph-factorized'](
                                                    N = N, 
                                                    M =192,
                                                    n_graph_encoder = n_graph_encoder,
                                                    symmetric = symmetric,
                                                    conv_type = conv_layer,
                                                    bipartite = bipartite,
                                                    cheb_k = cheb_k,
                                                    heads = heads,
                                                    activation = activation,
                                                    aggr = aggr,
                                                    knn = knn,
                                                    loop = loop,
                                                    use_ffn = use_ffn,
                                                    use_fc = use_fc,
                                                    graph_norm = graph_norm
                                                )
                                                """ 
                                                pytorch_total_params = sum(p.numel() for p in net.parameters())
                                                print('Total params: {:,}'.format(pytorch_total_params)) """

                                                
                                                net = net.to(device)

                                                out = net(x)
                                                outshape = out['x_hat'].shape
                                                print(f'Output shape: {outshape}')
                                                print('----------\n')