import torch.nn as nn
import torch 
import math
from pytorch_msssim import ms_ssim
from compressai.optimizers import net_aux_optimizer
import shutil
from gcn_lib import Grapher, FFN, Downsample, Upsample


def conv_graph(
        in_channels, 
        out_channels,
        conv='sage', # graph stuff
        cheb_k = 2,
        heads = 1,
        activation='none',
        aggr = 'mean',
        k=9,
        loop = True,
        ratio=[4,1], # graph stuff
        norm = 'none'): # conv2d stuff
    pass


def conv(
        in_channels, 
        out_channels,
        use_graph=False,
        bipartite = True,
        conv='sage', # graph stuff
        cheb_k = 2,
        heads = 1,
        activation='none',
        aggr = 'mean',
        k=9,
        loop = True,
        ratio=1, # graph stuff
        norm = 'none',
        use_ffn = False,
        use_fc = False,
        kernel_size=5, # conv2d stuff
        stride=2): # conv2d stuff

    if(use_graph):
        grapher = Grapher(
                in_channels=in_channels,
                out_channels=out_channels,
                knn=k,
                bipartite=bipartite,
                conv=conv,
                heads=heads,
                act=activation,
                aggr = aggr,
                norm=norm,
                bias=True,
                r=ratio,
                cheb_k=cheb_k,
                loop=loop,
                use_fc=use_fc)
        
        return nn.Sequential(
            grapher
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
        bipartite = True,
        conv='sage', # graph stuff
        cheb_k = 2,
        heads = 1,
        activation='none',
        aggr = 'mean',
        k=9,
        loop = True,
        ratio=1, # graph stuff
        norm = 'none',
        use_ffn = False,
        use_fc = False,
        kernel_size=5, # conv2d stuff
        stride=2): # conv2d stuff

        
    if(use_graph):
        grapher = Grapher(
                in_channels=in_channels,
                knn=k,
                bipartite=bipartite,
                conv=conv,
                heads=heads,
                act=activation,
                aggr = aggr,
                norm=norm,
                bias=True,
                r=ratio,
                cheb_k=cheb_k,
                loop=loop,
                use_fc=use_fc)
        ffn = FFN(
                in_features=in_channels,
                hidden_features=in_channels*4,
                out_features=in_channels,
                act=activation
            )
        up = Upsample(
                in_dim=in_channels,
                out_dim=out_channels
            )
        
        if(use_ffn):
            return nn.Sequential(
                grapher,
                ffn,
                up
            )
        else:
            return nn.Sequential(
                grapher,
                up
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