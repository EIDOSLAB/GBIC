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

from compressai.zoo import (
    bmshj2018_factorized,

)


image_models = {
    "graph-factorized":GBIC_FactorizedPrior,
    "bmshj2018-factorized": bmshj2018_factorized,

}





def main(argv):
    args = parse_args(argv)

    os.makedirs(args.save_dir, exist_ok=True)


    wandb.init(
        project='graph-compression',
        name=args.project_name,
        config={
            'model': args.model,
            'epochs':args.epochs,
            'batch_size':args.batch_size,
            'n_graph_encoder':args.n_graph_encoder,
            'symmetric':args.symmetric,
            'conv_layer':args.conv,
            'aggr':args.aggr,
            'graph_norm':args.graph_norm,
            'activation':args.activation,
            'knn':args.knn,
            'loop':args.loop,
            'use_ffn':args.use_ffn,
            'use_fc':args.use_fc,
            'Dataset_size': args.dataset_size,
            'N':args.N,
            'M':args.M,
            'lambda':args.lmbda
        }
    )


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
    device = "cuda" if  torch.cuda.is_available() else "cpu"

    N = args.N
    n_graph_encoder = args.n_graph_encoder
    symmetric = args.symmetric
    conv_layer_full = args.conv
    aggr = args.aggr
    graph_norm = args.graph_norm
    activation = args.activation
    loop = args.loop
    use_ffn = args.use_ffn
    use_fc = args.use_fc
    knn = args.knn

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


    if "graph" in args.model:
        net = image_models[args.model](
            N = N, 
            M = args.M,
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
            graph_norm = graph_norm)
    else:
        net = image_models[args.model](quality = args.quality)

    net = net.to(device)



    if torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    counter = 0
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
            counter
        )
        loss = test_one_epoch(epoch, test_dataloader, net, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        save_checkpoint(
            {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
            },
            is_best, out_dir = args.save_dir
        ) 
        

        if epoch%10==0:
            print("make actual compression")
            net.update(force = True)
            print("finito l'update")
            l = compress_one_epoch(net, test_dataloader, device,epoch_enc)
            epoch_enc += 1



if __name__ == "__main__":

    wandb.login()

    #wandb.init(project="prova", entity="albertopresta")

    main(sys.argv[1:])