import torch
from datetime import datetime
from utils import parser_arg, plot_accuracies, plot_losses
import joblib
from datasets import data_utils
from models import ResNet
from train import evaluate, fit
from PIL import Image
import matplotlib.pyplot as plt 
from pathlib import Path

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)

if __name__ == "__main__":
    # from datapath get classes
    
    args = parser_arg()
    opt = args
    opt.model_path = "./room_class_fication_7_class_04-16_19-09.h5"
    data_utils = data_utils(args)
    opt.device = get_default_device()
    if(opt.opt_func == "Adam"):
        opt_func = torch.optim.Adam

    #recog
    model = joblib.load(opt.model_path)
    img_name = "phone.jpg"
    while(True):
        img_name = input()
        image = Image.open(Path('./' + img_name))
        example_image = data_utils.transformations(image)
        # plt.imshow(example_image.permute(1, 2, 0))

        xb = to_device(torch.unsqueeze(data_utils.transformations(image),0), opt.device)
        yb = model(xb)
        prob, preds  = torch.max(yb, dim=1)
        pre_class = data_utils.dataset.classes[preds[0].item()]

        print("prediction class is:{}".format(pre_class))