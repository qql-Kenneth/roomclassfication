import torch
from datetime import datetime
from utils import parser_arg, plot_accuracies, plot_losses
import joblib
from datasets import data_utils
from models import ResNet
from train import evaluate, fit

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
    data_utils = data_utils(args)
    opt.classes = data_utils.get_class()
    print(opt.classes)
    print("len classes:{}".format(len(opt.classes)))

    train_dl, val_dl = data_utils.get_batch()

    model = ResNet(opt)

    opt.device = get_default_device()
    print("device:{}".format(opt.device))

    train_dl = DeviceDataLoader(train_dl, opt.device)
    val_dl = DeviceDataLoader(val_dl, opt.device)
    to_device(model, opt.device)

    model = to_device(model, opt.device)

    #test evaluate
    # print(evaluate(model, val_dl))

    if(opt.opt_func == "Adam"):
        opt_func = torch.optim.Adam

    #train 
    history = fit(opt.num_epochs, opt.lr, model, train_dl, val_dl, opt_func)
    
    # plot_accuracies(history)

    #evaluate
    evaluate(model, val_dl)
    current_time = datetime.now()
    formatted_time = current_time.strftime("%m-%d_%H-%M")
    joblib.dump(model, 'room_class_fication_{}_class_{}.h5'\
                .format(len(opt.classes),formatted_time),protocol=4)