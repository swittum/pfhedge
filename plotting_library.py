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

def save_training_diagram(history, path):
    plt.figure()
    plt.plot(history)
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.title("Loss history")
    plt.savefig(path)
