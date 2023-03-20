import torch 
import torch.nn as nn

from models.google import GBIC_FactorizedPrior, GBIC_FactorizedPriorReLU
from utils.functions import configure_optimizers, save_checkpoint, CustomDataParallel
from training.loss import RateDistortionLoss
from training.step import train_one_epoch, test_one_epoch, compress_one_epoch
from utils.parser import parse_args
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.dataset import VimeoDatasets, TestKodakDataset
import random
import sys
import wandb
import os
import argparse


from compressai.zoo import (
    bmshj2018_factorized,

)


image_models = {
    "graph-factorized":GBIC_FactorizedPrior,
    "bmshj2018-factorized": bmshj2018_factorized,

}

parser = argparse.ArgumentParser(description="Example training script.")
parser.add_argument("--td_path", type = str, help = "recovering dataset",default = "/scratch/dataset/vimeo_triplet/sequences" )
parser.add_argument("--file_txt",type = str, help = "dataset", default = "/scratch/dataset/vimeo_triplet/tri_trainlist.txt")
parser.add_argument("--test_pt", type = str, help = "test dataset", default = "/scratch/dataset/kodak")
parser.add_argument("--patch_size",type=int,nargs=2,default=(256, 256),help="Size of the patches to be cropped (default: %(default)s)",)
parser.add_argument("-nd","--dataset_size", help = "number of training images", type = int, default = 30000)
parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: %(default)s)")
parser.add_argument("-n","--num__workers",type=int,default=4,help="Dataloaders threads (default: %(default)s)",)
parser.add_argument("-e","--epochs",default=200,type=int,help="Number of epochs (default: %(default)s)",)
parser.add_argument("-lr","--learning-rate",default=1e-4,type=float,help="Learning rate (default: %(default)s)",)
parser.add_argument("--lambda",dest="lmbda",type=float,default=0.0018,help="Bit-rate distortion parameter (default: %(default)s)",)
parser.add_argument("-M","--M", type = int, default=192)
parser.add_argument("-N","--N", type = int, default=128)
parser.add_argument("--clip_max_norm",default=1.0,type=float,help="gradient clipping max norm (default: %(default)s",)
parser.add_argument("--aux-learning-rate",type=float,default=1e-3,help="Auxiliary loss learning rate (default: %(default)s)",)

parser.add_argument("--sweep-id", type = str, default = "none")

args = parser.parse_args()

def main():


    wandb.init(
        project = 'gnn-compression-sweep',
        name = 'q5/baseline',
        config={
            'model': 'baseline',
            'epochs':args.epochs,
            'batch_size':args.batch_size,
            'ConvType':'conv2d',
            'Dataset_size': args.dataset_size,
            'N':args.N,
            'M':args.M,
            'lambda':args.lmbda
        }
    )
    

    

    seed = 42
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)

    print("train set")
    full_dataset = VimeoDatasets(args.td_path, args.file_txt,args.patch_size, NUM_IMAGES= args.dataset_size) 
    #train_dataset = Datasets(td_path, img_size)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers= args.num__workers,
        shuffle=True,
        pin_memory=True,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers= args.num__workers,
        shuffle=False,
        pin_memory=True,
    )

    print("fine dataset")
    device = "cuda" if  torch.cuda.is_available() else "cpu"

    net = image_models['graph-factorized'](
        N =args.N, 
        M =args.M,
        n_graph_encoder = 0,
        symmetric = False,
    )

    net = net.to(device)

    if torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    counter = 0

    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            counter
        )
        loss = test_one_epoch(epoch, val_dataloader, net, criterion, tag='val')
        lr_scheduler.step(loss)

        
if __name__ == "__main__":

    wandb.login()
    main()