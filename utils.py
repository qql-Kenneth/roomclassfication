import matplotlib.pyplot as plt 
import argparse

def parser_arg():
    parser = argparse.ArgumentParser(description="Parser for room scence classficaiton!")
    parser.add_argument('--data_path', type=str,default='raw7data')
    parser.add_argument('--batch_size',type=int,default=12)
    parser.add_argument('--num_epochs',type=int,default=10)
    parser.add_argument('--opt_func',type=str,default="Adam")
    parser.add_argument('--lr',type=float,default=6e-5)
    return parser.parse_args()

def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');

def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');
