import torch 
import os 
import numpy as np 
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from utils.AverageMeter import AverageMeter
from compressai.ops import compute_padding
import math 
from pytorch_msssim import ms_ssim
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
from compressai.zoo import *
from utils.dataset import VimeoDatasets, TestKodakDataset
from torch.utils.data import DataLoader
from os.path import join 

from models.google import GBIC_FactorizedPrior

import sys

image_models = {
    "graph-factorized":GBIC_FactorizedPrior,

}



def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument("-m","--model",default="hyperprior",help="Model architecture (default: %(default)s)",)
    parser.add_argument("-mp","--model_path",default="/scratch/inference/pretrained_models",help="Model architecture (default: %(default)s)",)

    parser.add_argument("-rp","--result_path",default="/scratch/inference/results",help="Model architecture (default: %(default)s)",)

    parser.add_argument("-ip","--image_path",default="/scratch/dataset/kodak",help="Model architecture (default: %(default)s)",)



    args = parser.parse_args(argv)
    return args

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)




def reconstruct_image_with_nn(networks, filepath, device, save_path):
    reconstruction = {}
    for name, net in networks.items():
        #net.eval()
        with torch.no_grad():
            x = read_image(filepath).to(device)
            x = x.unsqueeze(0)
            out_net,= net(x,  False)
            out_net["x_hat"].clamp_(0.,1.)
            original_image = transforms.ToPILImage()(x.squeeze())
            reconstruction[name] = transforms.ToPILImage()(out_net['x_hat'].squeeze())




    svpt = os.path.joint(save_path,"original" + filepath.split("/")[-1])

    fix, axes = plt.subplots(1, 1)
    for ax in axes.ravel():
        ax.axis("off")

    axes.ravel()[0 ].imshow(original_image)
    axes.ravel()[0].title.set_text("original image")    
    
    plt.savefig(svpt)
    plt.close()


    svpt = os.path.joint(save_path,filepath.split("/")[-1])

    fix, axes = plt.subplots(5, 4, figsize=(10, 10))
    for ax in axes.ravel():
        ax.axis("off")
    

    for i, (name, rec) in enumerate(reconstruction.items()):
            #axes.ravel()[i + 1 ].imshow(rec.crop((468, 212, 768, 512))) # cropped for easy comparison
        axes.ravel()[i ].imshow(rec)
        axes.ravel()[i].title.set_text(name)

        #plt.show()
    plt.savefig(svpt)
    plt.close()



def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())


def compute_metrics( org, rec, max_val: int = 255):
    metrics =  {}
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    metrics["psnr"] = psnr(org, rec).item()
    metrics["ms-ssim"] = ms_ssim(org, rec, data_range=max_val).item()
    return metrics

def load_models(models_path,model_checkpoint, device, image_models):


    res = {}
    for model_check in model_checkpoint:
        model_path = join(models_path, model_check)
        checkpoint = torch.load(model_path, map_location=device)
        N = 160
        M = 192

        if('q6' in model_path or 'q7' in model_path or 'q8' in model_path ):
            N = 192
            M = 320

        model = image_models['graph-factorized'](
            N = N, 
            M = M,
            n_graph_encoder = 2,
            symmetric = False,
            conv_type = 'transformer',
            bipartite = True,
            cheb_k = 1,
            heads = 4,
            activation = 'none',
            aggr = 'max',
            knn = 15,
            loop = False,
            use_ffn = False,
            use_fc = False,
            graph_norm = 'none')
        model = model.to(device)
        model.load_state_dict(checkpoint["state_dict"],strict=False) 
        model.update()
        
        name = (model_path.split(os.sep)[-1]).replace('.pth.tar','')
        res[name] = {"model": model}
        print(f'{model_path} loaded')
    print()
    return res



def collect_images(rootpath: str):
    image_files = []

    for ext in IMG_EXTENSIONS:
        image_files.extend(Path(rootpath).rglob(f"*{ext}"))
    return sorted(image_files)


def read_image(filepath):
    assert filepath.is_file()
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)


@torch.no_grad()
def inference(model, dataloader, device):

    psnr = AverageMeter()
    ms_ssim = AverageMeter()
    bpps = AverageMeter()

    print("inizio inferenza")
    for i,d in enumerate(dataloader):
        #x = read_image(filepath).to(device)
        #print(i)
        d = d.to(device)
        #x = x.unsqueeze(0)
        #h, w = x.size(2), x.size(3)
        #pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
        #x_padded = F.pad(x, pad, mode="constant", value=0)
        #data =  model.compress(d)
        #if sos: 
        #    out_dec = model.decompress(data)
        #else:
        #    out_dec = model.decompress(data["strings"], data["shape"])
        #out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)
        #out_dec["x_hat"].clamp_(0.,1.)
        # input images are 8bit RGB for now
        #out_net  = model(d, training = False)
        #metrics = compute_metrics(d, out_net["x_hat"], 255)
        out_net  = model(d)
        metrics = compute_metrics(d, out_net["x_hat"], 255)

        size = out_net['x_hat'].size()
        num_pixels = size[0] * size[2] * size[3]
        y_bpp = torch.log(out_net["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)
        #z_bpp  = torch.log(out_net["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels)
        size = out_net['x_hat'].size() 
        num_pixels = size[0] * size[2] * size[3]
        
        bpp = y_bpp 

        psnr.update(metrics["psnr"])
        ms_ssim.update(metrics["ms-ssim"])
        bpps.update(bpp.item())
    print("fine inferenza",psnr.avg, ms_ssim.avg, bpps.avg)
    return psnr.avg, ms_ssim.avg, bpps.avg


def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


def bpp_calculation_factorized(out_net, out_enc):
        size = out_net['x_hat'].size() 
        num_pixels = size[0] * size[2] * size[3]
        bpp = (len(out_enc) * 8.0 ) / num_pixels
        #bpp_2 =  (len(out_enc[1]) * 8.0 ) / num_pixels

        return bpp

@torch.no_grad()
def eval_models(res, dataloader, device):
  
    metrics = {}
    models_name = list(res.keys())
    for i, name in enumerate(models_name): #name = q1-bmshj2018-base/fact
        print("----")
        print("name: ",name)
        model = res[name]["model"]

        psnr, mssim, bpp = inference(model,dataloader,device)

        metrics[name] = {
            "bpp": bpp,
            "mssim": mssim,
            "psnr": psnr
        }

        

    return metrics   




def extract_specific_model_performance(metrics, type):

    nms = list(metrics[type].keys())

    psnr = []
    mssim = []
    bpp = []
    for names in nms:
        psnr.append(metrics[type][names]["psnr"])
        mssim.append(metrics[type][names]["mssim"])
        bpp.append(metrics[type][names]["bpp"])
    
    return sorted(psnr), sorted(mssim), sorted(bpp)


Colors = {
    "graph-factorized": ["b",'--'],
    "graph-transf": ["g",'-'],
    "graph-mr": ["b",'-'],
    "baseline": ["r",'-'],
    "bmshj2017-sos": ["b","-"],
}

def plot_rate_distorsion(metrics, savepath, col = ["b","g","r","c","m","y","k"]):

    print(f'plotting on {savepath}')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plt.figtext(.5, 0., '(upper-left is better)', fontsize=12, ha='center')

    for type_name in metrics.keys():

        psnr, mssim, bpp = extract_specific_model_performance(metrics, type_name)   


        cols = Colors[type_name]      
        axes[0].plot(bpp, psnr,cols[1],color = cols[0], label = type_name)
        axes[0].plot(bpp, psnr,'o',color =  cols[0])


        axes[1].plot(bpp, mssim,cols[1],color =  cols[0], label = type_name)
        axes[1].plot(bpp, mssim,'o',color =  cols[0])


    axes[0].set_ylabel('PSNR [dB]')
    axes[0].set_xlabel('Bit-rate [bpp]')
    axes[0].title.set_text('PSNR comparison')
    axes[0].grid()
    axes[0].legend(loc='best')

 
    axes[1].set_ylabel('MS-SSIM [dB]')
    axes[1].set_xlabel('Bit-rate [bpp]')
    axes[1].title.set_text('MS-SSIM (log) comparison')
    axes[1].grid()
    axes[1].legend(loc='best')



    for ax in axes:
        ax.grid(True)
    plt.savefig(savepath)
    plt.close()    

     
from os import listdir

def produce_metrics(args):

    model_name = args['model']  # nome del modello che voglio studiare (ad esempio cheng2020)
    models_path = join(args['model_path'],model_name) # percorso completo per arrivare ai modelli salvati (/scratch/inference/pretrained_models/chegn2020) qua ho salvato i modelli 
 
    models_checkpoint = []
    for entry in os.listdir(models_path):
        if('pth.tar' in entry):
            models_checkpoint.append(entry) # checkpoints dei modelli  q1-bmshj2018-sos.pth.tar, q2-....
    print(models_checkpoint)
    device = "cuda"

    
    images_path = args['image_path'] # path del test set 


    
    test_dataset = TestKodakDataset(data_dir= images_path ) #test set 
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=4) # data loader 



    res = load_models(models_path,models_checkpoint, device, image_models) # carico i modelli res è un dict che ha questa struttura res[q1-bmshj2018-sos] = {"model": model}
    

    

    metrics = eval_models(res, test_dataloader , device) #faccio valutazione dei modelli 



    #metrics ha questa struttura         metrics[name] = {"bpp": bpp,
    #                        "mssim": mssim,
    #                        "psnr": psnr
    #                         }

    return metrics
    list_names = list(image_models.keys())   
    plot_rate_distorsion(metrics,list_names) # plot del rate distorsion 


if __name__ == "__main__":
    
    my_parser = argparse.ArgumentParser(description= "path to read the configuration of the evaluation")
    my_parser.add_argument("-c","--config", default="configuration/inference.json", type=str,
                      help='config file path')
    my_parser.add_argument("--metrics", default="none", type=str,
                      help='metrics json file')
    my_parser.add_argument("--model-tag", default="none", type=str,
                      help='Tag model')
    
    args = my_parser.parse_args()
    
    config_path = args.config

    new_metrics = {}
    if(args.metrics == 'none'):
    

        with open(config_path) as f:
            config = json.load(f)

        #print(config)
        metrics = produce_metrics(config)

        new_metrics = {}
        new_metrics[args.model_tag] = {}
        for model in metrics.keys():
            q = model.split('-')[0]
            new_metrics[args.model_tag][q] = metrics[model]
        
        file_path = join(config['model_path'],config['model'],'metrics.json')
        with open(file_path, 'w') as outfile:
            json.dump(new_metrics, outfile)

        save_path = join(config['model_path'],config['model'],'metrics.png')
    else:
        work_path = '/scratch/GBIC_res/eval/'

        with open(args.metrics) as json_file:
            new_metrics = json.load(json_file)
        save_path = join(work_path,'metrics.png')
    

    plot_rate_distorsion(new_metrics,save_path)

    
