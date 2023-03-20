import torch
import wandb
from utils.AverageMeter import AverageMeter
from utils.functions import compute_psnr, compute_msssim, compute_metrics
import torch.nn.functional as F
from compressai.ops import compute_padding




def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, counter):
    model.train()
    device = next(model.parameters()).device
    
    
    
    loss_tot = AverageMeter()
    bpp = AverageMeter()
    mse = AverageMeter()
    aux = AverageMeter()




    for i, d in enumerate(train_dataloader):

        d = d.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        out_net = model(d)      
        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()


        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()



        if i % 10 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )
        # log on wandb     
        wand_dict = {
            "train_batch": counter,
            "train_batch/losses_batch": out_criterion["loss"].clone().detach(),
            "train_batch/bpp_batch": out_criterion["bpp_loss"].clone().detach(),
            "train_batch/mse":out_criterion["mse_loss"].clone().detach(),
            "train_batch/aux_loss":aux_loss.clone().detach()

        }
        wandb.log(wand_dict)
        counter += 1
        
        
    loss_tot.update(out_criterion["loss"].clone().detach())
    bpp.update(out_criterion["bpp_loss"].clone().detach())
    mse.update(out_criterion["mse_loss"].clone().detach())
    aux.update(aux_loss.clone().detach())
    
        
    log_dict = {
        "train":epoch,
        "train/loss": loss_tot.avg,
        "train/bpp": bpp.avg,
        "train/mse": mse.avg,
        "train/aux_loss":aux.avg
    }
        
    wandb.log(log_dict)      
        
    return counter 

    


def test_one_epoch(epoch, test_dataloader, model, criterion, tag='test'):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()
    ssim = AverageMeter()


    with torch.no_grad():
        for i,d in enumerate(test_dataloader):
            d = d.to(device)
            out_net = model(d)
                        
            out_criterion = criterion(out_net, d)


            psnr.update(compute_psnr(d, out_net["x_hat"]))
            ssim.update(compute_msssim(d, out_net["x_hat"]))
            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])


    print(
        f"{tag} epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    log_dict = {
    f"{tag}":epoch,
    f"{tag}/loss": loss.avg,
    f"{tag}/bpp":bpp_loss.avg,
    f"{tag}/mse": mse_loss.avg, 
    f"{tag}/psnr":psnr.avg,
    f"{tag}/ssim":ssim.avg,
    }
    wandb.log(log_dict)
    return loss.avg



def compress_one_epoch(model, test_dataloader, device, epoch):
    #model.update(None, device)
    print("ho finito l'update")
    bpp_loss = AverageMeter()
    psnr = AverageMeter()
    mssim = AverageMeter()

    
    with torch.no_grad():
        for i,d in enumerate(test_dataloader): 
            print("-------------    immagine ",i,"  --------------------------------")
            d = d.to(device)
            h, w = d.size(2), d.size(3)
            pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2

            x_padded = F.pad(d, pad, mode="constant", value=0)
            out_enc = model.compress(x_padded)
            out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
            out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)
            metrics = compute_metrics(d, out_dec["x_hat"], 255)
            num_pixels = d.size(0) * d.size(2) * d.size(3)
            bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
            
            psnr.update( metrics["psnr"])
            mssim.update( metrics["ms-ssim"]) 
            bpp_loss.update(bpp)  



    log_dict = {
            "test":epoch,
            "test/bpp_with_ac": bpp_loss.avg,
            "test/psnr_with_ac": psnr.avg,
            "test/mssim_with_ac":mssim.avg
    }
    
    wandb.log(log_dict)
    return bpp_loss.avg





def bpp_calculation_factorized(out_net, out_enc):
        size = out_net['x_hat'].size() 
        num_pixels = size[0] * size[2] * size[3]
        bpp = (len(out_enc) * 8.0 ) / num_pixels
        #bpp_2 =  (len(out_enc[1]) * 8.0 ) / num_pixels

        return bpp