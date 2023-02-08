import matplotlib.pyplot as plt
import numpy as np
import torch
from io import BytesIO
import base64
import pandas as pd

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

def save_multi_profit(profits,params,path, bench={}):
    data = pd.DataFrame({'params':params,'shortfall':profits})
    avg = data.groupby('params').mean()
    plt.figure()
    plt.xscale('log')
    plt.plot(avg)
    plt.xlabel("Number of parameters")
    plt.ylabel("Expected Shortfall")
    plt.title("Performance by number of parameters")
    if 'WW' in bench.keys():
        plt.axhline(bench['WW'], color='b')
    if 'No' in bench.keys():
        plt.axhline(bench['No'], color='r')
    plt.savefig(path)
    plt.close()
