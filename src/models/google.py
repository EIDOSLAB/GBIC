import warnings
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

#from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck   , GaussianConditional
from compressai.layers import GDN  #, MaskedConv2d
from compressai.registry import register_model

from compressai.models.base import (
    CompressionModel,
 
)
from utils.functions import conv, deconv

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

    def __init__(self, N, M,use_graph_encoder = False,use_graph_decoder = False, conv_type='mr', **kwargs):
        super().__init__(**kwargs)

        self.entropy_bottleneck = EntropyBottleneck(M)

        self.g_a = nn.Sequential(
            conv(3, N, use_graph=False),
            GDN(N),
            #conv(N, N,use_graph=use_graph_encoder,conv=conv_type, ratio=8),
            conv(N, N),
            GDN(N),
            conv(N, N,use_graph=use_graph_encoder,conv=conv_type, ratio=4),
            #conv(N, N),
            GDN(N),
            conv(N, M,use_graph=use_graph_encoder,conv=conv_type, ratio=1),
            #conv(M, M,use_graph=use_graph_encoder,conv=conv_type, reduce_graph=False, ratio=1)
        )

        self.g_s = nn.Sequential(
            deconv(M, N,use_graph=use_graph_decoder,conv=conv_type, ratio=1),
            GDN(N, inverse=True),
            deconv(N, N,use_graph=use_graph_decoder,conv=conv_type, ratio=4),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
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


@register_model("graphbased-scalehyperprior")
class  GBIC_ScaleHyperprior(CompressionModel):



    def __init__(self, N, M, **kwargs):
        super().__init__(**kwargs)

        self.entropy_bottleneck = EntropyBottleneck(N)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
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
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}






if __name__ == '__main__':


    image_models = {
        "graph-factorized":GBIC_FactorizedPrior,
        "bmshj2018-factorized": bmshj2018_factorized,
    }   
    device = "cuda" if  torch.cuda.is_available() else "cpu"

    net = image_models['graph-factorized'](
        N = 128, 
        M =192,
        use_graph_encoder = True,
        use_graph_decoder = False, 
        conv_type='mr')
    
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print('Vanilla: total params: {:,}'.format(pytorch_total_params))

    
    net = net.to(device)

    x = torch.rand((8,3,256,256))
    x= x.to(device)
    st = time.time()
    out = net(x)
    print(f'Time: {time.time()-st}')
    outshape = out['x_hat'].shape
    print(f'Output shape: {outshape}')