from io import BytesIO
import base64
import matplotlib.pyplot as plt
import numpy as np
import torch
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
    plt.close(figure)
    fig.seek(0)
    imgstring = base64.b64encode(fig.getvalue())
    return imgstring.decode('utf-8')

def make_multi_profit(profits,params, bench={}):
    data = pd.DataFrame({'params':params,'shortfall':profits})
    avg = data.groupby('params').mean()
    fig = plt.figure()
    plt.xscale('log')
    plt.plot(avg)
    plt.xlabel("Number of parameters")
    plt.ylabel("Expected Shortfall")
    plt.title("Performance by number of parameters")
    colors_dict = {'Whalley-Wilmott': 'b', 'Black-Scholes': 'g', 'No Hedge': 'r'}
    for key,value in bench.items():
        plt.axhline(value,color=colors_dict[key])
    return fig

def make_stock_diagram(listings):
    fig = plt.figure()
    plt.plot(to_numpy(listings))
    plt.xlabel("Business day", fontsize=20)
    plt.ylabel("Stock price", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    return fig
