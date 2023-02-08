import matplotlib.pyplot as plt
import numpy as np
import torch
from io import BytesIO
import base64

def to_numpy(tensor: torch.Tensor) -> np.array:
    return tensor.cpu().detach().numpy()

def make_pl_diagram(pnl):
    figure = plt.figure()
    plt.hist(to_numpy(pnl), bins=100)
    plt.title(f"Profit-loss histogram of {torch.numel(pnl)} price paths")
    plt.xlabel("Profit-loss")
    plt.ylabel("Number of events")
    return figure

def make_training_diagram(history):
    figure = plt.figure()
    plt.plot(history)
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.title("Loss history")
    return figure

def figure_to_string(figure):
    fig = BytesIO()
    figure.savefig(fig,format='png')
    fig.seek(0)
    imgstring = base64.b64encode(fig.getvalue())
    return imgstring.decode('utf-8')