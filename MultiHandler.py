from typing import List
import matplotlib.pyplot as plt
from HedgeHandler import HedgeHandler
from plotting_library import make_multi_profit
class MultiHandler:
    def __init__(self, handlers: List[HedgeHandler], n_params: List[int]):
        self.handlers = handlers
        self.params = n_params
    def fit(self):
        for handler in self.handlers:
            handler.fit()
    def profit(self) -> List[float]:
        return [handler.eval(handler.profit()) for handler in self.handlers]
    def benchmark(self):
        handler = self.handlers[-1]
        return handler.dict_eval(handler.benchmark())
    def full_process(self):
        self.fit()
        bench = self.benchmark()
        profits = self.profit()
        fig = make_multi_profit(profits,self.params, bench)
        fig.savefig('paramsprofits.png')
        plt.close(fig)