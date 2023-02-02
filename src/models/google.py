import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

#from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck   #, GaussianConditional
from compressai.layers import GDN  #, MaskedConv2d
from compressai.registry import register_model

from compressai.models.base import (
    CompressionModel,
 
)
from utils.functions import conv, deconv


__all__ = [
    "CompressionModel",
    "FactorizedPrior",
    "FactorizedPriorReLU"
]



@register_model("graphbased-factorized")
class GBIC_FactorizedPrior(CompressionModel):
    r"""ADD GRAPH MODULES
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(**kwargs)

        self.entropy_bottleneck = EntropyBottleneck(M)

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
