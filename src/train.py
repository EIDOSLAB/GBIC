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


from compressai.image import (
    bmshj2018_factorized,

)


image_models = {
    "graph-factorized":GBIC_FactorizedPrior,
    "bmshj2018-factorized": bmshj2018_factorized,



}





def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    print("train set")
    train_dataset = VimeoDatasets(args.td_path, args.file_txt,args.patch_size, NUM_IMAGES= args.dataset_size) 
    #train_dataset = Datasets(td_path, img_size)
    

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers= args.num__workers,
        shuffle=True,
        pin_memory=True,
    )
    

    
    test_dataset = TestKodakDataset(data_dir= args.test_pt)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers= args.num__workers )




    print("fine dataset")
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"



    net = image_models[args.model](N = args.N, M = args.M)

    net = net.to(device)



    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint != "_":  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    epoch_enc = 0
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
        )
        loss = test_one_epoch(epoch, test_dataloader, net, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if is_best or epoch%10==0:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
            )
        

        if epoch%10==0:
            print("make actual compression")
            net.update(force = True)
            print("finito l'update")
            l = compress_one_epoch(net, test_dataloader, device,epoch_enc)
            epoch_enc += 1



if __name__ == "__main__":
    main(sys.argv[1:])