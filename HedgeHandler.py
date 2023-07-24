from typing import List
import matplotlib.pyplot as plt
from pfhedge.instruments import BaseDerivative, BaseInstrument
from pfhedge.nn import Hedger, HedgeLoss, WhalleyWilmott, BlackScholes, Naked
from plotting_library import make_training_diagram, make_pl_diagram, make_stock_diagram
class HedgeHandler:
    def __init__(self, hedger: Hedger, derivative: BaseDerivative, hedge: List[BaseInstrument], fit_params: dict, profit_params: dict, criterion: HedgeLoss, benchmark_params: dict):
        self.hedger = hedger
        self.derivative = derivative
        self.hedge = hedge
        self.fit_params = fit_params
        self.profit_params = profit_params
        self.criterion = criterion
        self.benchmark_params = benchmark_params
    def fit(self):
        return self.hedger.fit(self.derivative,self.hedge,**self.fit_params)
    def profit(self):
        return self.hedger.compute_pnl(self.derivative,self.hedge,**self.profit_params)
    def benchmark(self):
        output = {}
        if self.benchmark_params.get('WhalleyWilmott',False):
            compmodel = WhalleyWilmott(self.derivative)
            comphedger = Hedger(compmodel, inputs=compmodel.inputs())
            comp = comphedger.compute_pnl(self.derivative, **self.profit_params)
            output['Whalley-Wilmott'] = comp
        if self.benchmark_params.get('BlackScholes', False):
            compmodel = BlackScholes(self.derivative)
            comphedger = Hedger(compmodel, inputs=compmodel.inputs())
            comp = comphedger.compute_pnl(self.derivative, **self.profit_params)
            output['Black-Scholes'] = comp
        if self.benchmark_params.get('NoHedge',False):
            nohedger = Hedger(Naked(),inputs=["empty"])
            nohedge = nohedger.compute_pnl(self.derivative,**self.profit_params)
            output['No Hedge'] = nohedge
        return output
    def eval(self,pnl):
        return self.criterion(pnl).item()
    def dict_eval(self,dictionary):
        output = {}
        for key in dictionary.keys():
            output[key] = self.eval(dictionary[key])
        return output

    def full_process(self, backup):
        if backup:
            print('Loading backup.')
            self.hedger.load_backup('./Backup_Parameters/params.pt') 
        else:
            history = self.fit()
            training_fig = make_training_diagram(history)
            training_fig.savefig('trainingdiagram.png', bbox_inches='tight')
            plt.close(training_fig)

        pnl = self.profit()
        bench = self.benchmark()

        # import json
        # with open('./output.txt', 'w') as ofile:
        #     ofile.write(str(self.eval(pnl)))
        #     ofile.write('\n')
        #     ofile.write(json.dumps(self.dict_eval(bench)))
        #     ofile.write('\n')

        print(self.eval(pnl))
        print(self.dict_eval(bench))

        pnl_fig = make_pl_diagram(pnl)
        pnl_fig.savefig('pldiagram.png', bbox_inches='tight')
        plt.close(pnl_fig)
        for key, value in bench.items():
            fig = make_pl_diagram(value)
            fig.savefig(f'pl{(key[0:2].lower())}')
            plt.close(fig)
    def stock_diagrams(self, number):
        self.derivative.simulate(number)
        prices = self.derivative.underlier.spot
        for i in range(number):
            fig = make_stock_diagram(prices[i,...])
            fig.savefig(f'stockdiagram{i}', bbox_inches='tight')
            plt.close(fig)

        

        