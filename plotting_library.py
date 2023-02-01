import matplotlib.pyplot as plt
import numpy as np
import torch

def to_numpy(tensor: torch.Tensor) -> np.array:
    return tensor.cpu().detach().numpy()

def save_pl_diagram(pnl, path):
    plt.figure()
    plt.hist(to_numpy(pnl), bins=100)
    plt.title(f"Profit-loss histogram of {torch.numel(pnl)} price paths")
    plt.xlabel("Profit-loss")
    plt.ylabel("Number of events")
    plt.savefig(path)
    plt.close()

def save_training_diagram(history, path):
    plt.figure()
    plt.plot(history)
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.title("Loss history")
    plt.savefig(path)
    plt.close()

def save_multi_profit(profits,params,path, bench={}):
    plt.figure()
    plt.xscale('log')
    plt.plot(params,profits)
    plt.xlabel("Number of parameters")
    plt.ylabel("Expected Shortfall")
    plt.title("Performance by number of parameters")
    if 'WW' in bench.keys():
        plt.axhline(bench['WW'], color='b')
    if 'No' in bench.keys():
        plt.axhline(bench['No'], color='r')
    plt.savefig(path)
    plt.close()