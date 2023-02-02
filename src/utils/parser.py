import argparse



def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument("-m","--model",default="graph-factorized",help="Model architecture (default: %(default)s)",type = str)
    parser.add_argument("-N","--N", type = int, default = 128)
    parser.add_argument("-M","--M", type = int, default=192)

    parser.add_argument("-e","--epochs",default=10,type=int,help="Number of epochs (default: %(default)s)",)
    parser.add_argument("-lr","--learning-rate",default=1e-4,type=float,help="Learning rate (default: %(default)s)",)
    parser.add_argument("-n","--num__workers",type=int,default=4,help="Dataloaders threads (default: %(default)s)",)
    parser.add_argument("--lambda",dest="lmbda",type=float,default=0.0018,help="Bit-rate distortion parameter (default: %(default)s)",)
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: %(default)s)")
    parser.add_argument("--test-batch-size",type=int,default=64,help="Test batch size (default: %(default)s)",)
    parser.add_argument("--aux-learning-rate",type=float,default=1e-3,help="Auxiliary loss learning rate (default: %(default)s)",)
    parser.add_argument("--patch_size",type=int,nargs=2,default=(256, 256),help="Size of the patches to be cropped (default: %(default)s)",)
    parser.add_argument("--save", action="store_true", default=True, help="Save model to disk")
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility", default = 42)
    parser.add_argument("--clip_max_norm",default=1.0,type=float,help="gradient clipping max norm (default: %(default)s",)
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint", default = "_")
    parser.add_argument("-nd","--dataset_size", help = "number of training images", type = int, default = 3000)

    parser.add_argument("--td_path", type = str, help = "recovering dataset",default = "/scratch/dataset/vimeo_triplet/sequences" )
    parser.add_argument("--file_txt",type = str, help = "dataset", default = "/scratch/dataset/vimeo_triplet/tri_trainlist.txt")
    parser.add_argument("--test_pt", type = str, help = "test dataset", default = "/scratch/dataset/kodak")

    args = parser.parse_args(argv)
    return args