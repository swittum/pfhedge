from pfhedge.instruments import BaseDerivative, BaseInstrument
from typing import List
from pfhedge.nn import Hedger, HedgeLoss, WhalleyWilmott, Naked
from plotting_library import save_training_diagram, save_pl_diagram
class HedgeHandler:
    def __init__(self, hedger: Hedger, derivative: BaseDerivative, hedge: List[BaseInstrument], fit_params: dict, profit_params: dict, criterion: HedgeLoss, benchmark_params: dict):
        self.hedger = hedger
        self.derivative = derivative
        self.hedge = hedge
        self.fit_params = fit_params
        self.profit_params = profit_params
        self.criterion = criterion
        self.benchmark_params = benchmark_params
    def fit(self, plot=True):
        history = self.hedger.fit(self.derivative,self.hedge,**self.fit_params)
        if plot:
            save_training_diagram(history, 'trainingdiagram.png')
    def profit(self,plot=True):
        pnl = self.hedger.compute_pnl(self.derivative,self.hedge,**self.profit_params)
        loss = self.criterion(pnl).item()
        print(loss)
        if plot:
            save_pl_diagram(pnl, 'pldiagram.png')
        return loss
    def benchmark(self):
        output = {}
        if self.benchmark_params.get('WhalleyWilmott',False):
            compmodel = WhalleyWilmott(self.derivative) 
            comphedger = Hedger(compmodel, inputs=compmodel.inputs())
            print("Whalley-Wilmott:")
            comp = comphedger.compute_pnl(self.derivative, **self.profit_params)
            save_pl_diagram(comp,'plww.png')
            output['WW'] = self.criterion(comp).item()
            print(output['WW'])
        if self.benchmark_params.get('NoHedge',False):
            nohedger = Hedger(Naked(),inputs=["empty"])
            print("No hedge:")
            nohedge = nohedger.compute_pnl(self.derivative,**self.profit_params)
            output['No'] = self.criterion(nohedge).item()
            print(output['No'])
            save_pl_diagram(nohedge,'plnone.png')
        return output

        