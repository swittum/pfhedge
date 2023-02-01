from typing import List
from HedgeHandler import HedgeHandler
from plotting_library import save_multi_profit
class MultiHandler:
    def __init__(self, handlers: List[HedgeHandler], n_params: List[int]):
        self.handlers = handlers
        self.params = n_params
    def fit(self):
        for handler in self.handlers:
            handler.fit(False)
    def profit(self, bench:dict = {}):
        profits = [handler.profit(False) for handler in self.handlers]
        save_multi_profit(profits,self.params,'paramsprofits.png', bench)
    def benchmark(self):
        return self.handlers[-1].benchmark()