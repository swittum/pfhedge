from pfhedge.instruments import BaseDerivative, BaseInstrument
from typing import List
from pfhedge.nn import Hedger, HedgeLoss, WhalleyWilmott, Naked
from plotting_library import make_training_diagram, make_pl_diagram
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
        if self.benchmark_params.get('NoHedge',False):
            nohedger = Hedger(Naked(),inputs=["empty"])
            print("No hedge:")
            nohedge = nohedger.compute_pnl(self.derivative,**self.profit_params)
            output['No Hedge'] = nohedge
        return output
    def eval(self,pnl):
        return self.criterion(pnl).item()
    def dict_eval(self,dict):
        output = {}
        for key in dict.keys():
            output[key] = self.eval(dict[key])
        return output
    def full_process(self):
        history = self.fit()
        pnl = self.profit()
        bench = self.benchmark()
        print(self.eval(pnl))
        print(self.dict_eval(bench))
        training_fig = make_training_diagram(history)
        training_fig.savefig('trainingdiagram.png')
        pnl_fig = make_pl_diagram(pnl)
        pnl_fig.savefig('pldiagram.png')
        if "Whalley-Wilmott" in bench.keys():
            ww_fig = make_pl_diagram(bench['Whalley-Wilmott'])
            ww_fig.savefig('plww.png')
        if "No Hedge" in bench.keys():
            no_fig = make_pl_diagram(bench['No hedge'])
            no_fig. savefig('plnone.png')

        