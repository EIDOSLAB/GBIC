import torch.nn as nn
import torch 
import math
from pytorch_msssim import ms_ssim
from compressai.optimizers import net_aux_optimizer
import shutil
from gcn_lib import Grapher, FFN, Downsample, Upsample

def conv(
        in_channels, 
        out_channels,
        use_graph=False,
        conv='mr', # graph stuff
        ratio=1, # graph stuff
        reduce_graph=True,
        kernel_size=5, # conv2d stuff
        stride=2): # conv2d stuff

    if(use_graph):
        if(reduce_graph):
            return nn.Sequential(
                Grapher(
                    in_channels=in_channels,
                    knn=9, 
                    dilation=1,
                    conv=conv,
                    heads=1,
                    act=None,
                    norm=None,
                    bias=True,
                    stochastic=False,
                    epsilon=0.0,
                    r=ratio,
                    relative_pos=False),
                #FFN(
                #    in_features=in_channels,
                #    hidden_features=in_channels*4,
                #    out_features=in_channels,
                #    act=None
                #),
                Downsample(
                    in_dim=in_channels,
                    out_dim=out_channels
                )
            )
        return nn.Sequential(
                Grapher(
                    in_channels=in_channels,
                    knn=9, 
                    dilation=1,
                    conv=conv,
                    heads=1,
                    act=None,
                    norm=None,
                    bias=True,
                    stochastic=False,
                    epsilon=0.0,
                    r=ratio,
                    relative_pos=False)
        )
    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(
        in_channels, 
        out_channels, 
        use_graph=False,
        conv='mr', # graph stuff
        ratio=1, # graph stuff
        kernel_size=5, # conv2d stuff
        stride=2): # conv2d stuff

    if(use_graph):
        return nn.Sequential(
            Grapher(
                in_channels=in_channels,
                knn=9, 
                dilation=1,
                conv=conv,
                heads=1,
                act=None,
                norm=None,
                bias=True,
                stochastic=False,
                epsilon=0.0,
                r=ratio,
                relative_pos=False),
            #FFN(
            #    in_features=in_channels,
            #    hidden_features=in_channels*4,
            #    out_features=in_channels,
            #    act=None
            #),
            Upsample(
                in_dim=in_channels,
                out_dim=out_channels
            )
        )

    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    ) 
    



def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": args.aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]


def save_checkpoint(state, is_best, out_dir, filename='checkpoint.pth.tar'):
    
    torch.save(state, f'{out_dir}/{filename}')
    if is_best:
        shutil.copyfile(f'{out_dir}/{filename}', f'{out_dir}/model_best.pth.tar')



def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255):
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())

def compute_metrics(org, rec, max_val = 255):
    metrics = {}
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    metrics["psnr"] = psnr(org, rec).item()
    metrics["ms-ssim"] = ms_ssim(org, rec, data_range=max_val).item()
    return metrics