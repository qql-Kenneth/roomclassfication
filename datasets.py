import os
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader 
from torch.utils.data import random_split 
import torch

class data_utils():
    def __init__(self,args):
        self.data_dir = args.data_path
        self.batch_size = args.batch_size
        self.classes = os.listdir(self.data_dir)
        self.transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        self.dataset = ImageFolder(self.data_dir, transform=self.transformations)

    def get_batch(self):
        random_seed = 21 
        torch.manual_seed(random_seed)
        train_ds, val_ds, test_ds = random_split(self.dataset, [2400,400,228])
        train_dl = DataLoader(train_ds, self.batch_size, shuffle = True, num_workers = 4, pin_memory = True)
        val_dl = DataLoader(val_ds, self.batch_size*2, num_workers = 4, pin_memory = True)
        return train_dl, val_dl
    
    def get_class(self):
        return self.classes
    
    def get_sample(self,img_num):
        img, label = self.dataset[img_num]
        return img,label