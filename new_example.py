from models import MultiLayerHybrid, NoTransactionBandNet
from quantum_circuits import SimpleQuantumCircuit
from plotting_library import save_pl_diagram
from pfhedge.nn import Hedger, ExpectedShortfall, MultiLayerPerceptron
from pfhedge.instruments import BrownianStock, HestonStock
from pfhedge.instruments import EuropeanOption

if __name__ == "__main__":
    n_qubits=2
    features = ["log_moneyness", "expiry_time", "volatility", "prev_hedge"]
    QUANTUM=True
    NTB = 1
    if QUANTUM:
        TwoQubit = SimpleQuantumCircuit(n_qubits,4)
        model = MultiLayerHybrid(TwoQubit.get_module(),n_qubits,n_qubits,n_layers=1,n_units=2, out_features=NTB+1)
    else:
        model = MultiLayerPerceptron(out_features=NTB+1)
    stock = BrownianStock(cost=1e-4)
    derivative = EuropeanOption(stock)
    if NTB==1:
        model = NoTransactionBandNet(derivative,model)
    expected_shortfall = ExpectedShortfall(0.1)
    hedger = Hedger(model,inputs=features,criterion=expected_shortfall)
    if QUANTUM:
        hedger.fit(derivative,n_epochs=6,n_paths=200,n_times=3)
    else:
        hedger.fit(derivative,n_epochs=50,n_paths=5000,n_times=10)
    pnl = hedger.compute_pnl(derivative,n_paths=500)
    save_pl_diagram(pnl, 'pldiagram.png')