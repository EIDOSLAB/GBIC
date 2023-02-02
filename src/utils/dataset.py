from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile

import os
from glob import glob
from torchvision import transforms
from torch.utils.data.dataset import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True







class VimeoDatasets(Dataset):
    def __init__(self, data_dir, text_dir , image_size=256, NUM_IMAGES = 3000):
        self.text_dir  = text_dir # tri_trainlist.txt
        self.data_dir = data_dir #vimeo_arod/
        self.image_size = image_size
        self.image_path = [] #sorted(glob(os.path.join(self.data_dir, "*.*")))
        self.total_dir = os.path.join(self.data_dir,"sequences") #data_dir/sequences
        
        file = open(os.path.join(self.data_dir,self.text_dir),"r")
        lines = file.readlines()
        
        for index, line in enumerate(lines):

            if index > NUM_IMAGES + 1:
                break
            c = line.strip()
            tmp = os.path.join(self.data_dir,c)
            d = [os.path.join(tmp,f) for f in os.listdir(tmp)]
            self.image_path += d
        
        file.close()
        self.image_path = self.image_path[:NUM_IMAGES]
        print("lunghezza del dataset: ",len(self.image_path))
 
        #self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))
        
    def __getitem__(self, item):
        image_ori = self.image_path[item]

        #image = cv2.imread(image_ori)

        image = Image.open(image_ori).convert('RGB')

        transform = transforms.Compose(
            [transforms.RandomCrop(self.image_size), transforms.ToTensor()]
        )
        return transform(image)
        
    def __len__(self):
        return len(self.image_path)












class TestKodakDataset(Dataset):
    def __init__(self, data_dir, image_size = 256):
        self.data_dir = data_dir
        self.image_size = image_size 
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        transform = transforms.Compose(
        [transforms.CenterCrop(self.image_size), transforms.ToTensor()])
        return transform(image)

    def __len__(self):
        return len(self.image_path)

