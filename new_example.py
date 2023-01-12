from models import MultiLayerHybrid, NoTransactionBandNet
from quantum_circuits import SimpleQuantumCircuit
from plotting_library import save_pl_diagram, save_training_diagram
from pfhedge.nn import Hedger, ExpectedShortfall, MultiLayerPerceptron, BlackScholes, WhalleyWilmott, Naked
from pfhedge.instruments import BrownianStock, HestonStock
from pfhedge.instruments import EuropeanOption, EuropeanBinaryOption, AmericanBinaryOption, LookbackOption
import seaborn


if __name__ == "__main__":
    seaborn.set_style("whitegrid")
    n_qubits=2
    features = ["log_moneyness", "expiry_time", "volatility","prev_hedge"]
    QUANTUM=True
    NTB = 1
    if QUANTUM:
        TwoQubit = SimpleQuantumCircuit(n_qubits,4)
        model = MultiLayerHybrid(TwoQubit.get_module(),n_qubits,n_qubits,n_layers=1,n_units=2, out_features=NTB+1)
    else:
        model = MultiLayerPerceptron(out_features=NTB+1)
    stock = BrownianStock(cost=5e-4)
    derivative = EuropeanOption(stock)
    if derivative.__class__ == AmericanBinaryOption or derivative.__class__ == LookbackOption:
        features = ["log_moneyness", "max_log_moneyness", "expiry_time", "volatility", "prev_hedge"]

    if NTB==1:
        model = NoTransactionBandNet(derivative,model)
    expected_shortfall = ExpectedShortfall(0.1)
    hedger = Hedger(model,inputs=features,criterion=expected_shortfall)
    if QUANTUM:
        history = hedger.fit(derivative,n_epochs=10,n_paths=250,n_times=1)
    else:
        history = hedger.fit(derivative,n_epochs=100,n_paths=5000,n_times=8)
    pnl = hedger.compute_pnl(derivative,n_paths=1000)
    save_pl_diagram(pnl, 'pldiagram.png')
    save_training_diagram(history,'trainingdiagram.png')
    compmodel = WhalleyWilmott(derivative)
    comphedger = Hedger(compmodel, inputs=compmodel.inputs())
    #comphedger = Hedger(Naked(),inputs=["empty"])
    comp = comphedger.compute_pnl(derivative, n_paths=1000)
    print(expected_shortfall(pnl))
    print(expected_shortfall(comp))