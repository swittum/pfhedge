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
    plt.title(f"Profit-loss histogram of {torch.numel(pnl)} price paths", fontsize=20)
    plt.xlabel("Profit-loss", fontsize=16)
    plt.ylabel("Number of events", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    return figure

def make_training_diagram(history):
    figure = plt.figure()
    plt.plot(history)
    plt.xlabel("Number of epochs", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.title("Loss history", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    return figure

def figure_to_string(figure):
    fig = BytesIO()
    figure.savefig(fig,format='png', bbox_inches='tight')
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
    plt.xlabel("Number of parameters", fontsize=20)
    plt.ylabel("Expected Shortfall", fontsize=20)
    plt.title("Performance by number of parameters", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
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
