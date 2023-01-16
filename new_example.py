from models import MultiLayerHybrid, NoTransactionBandNet
from quantum_circuits import SimpleQuantumCircuit
from plotting_library import save_pl_diagram, save_training_diagram
from utils import black_scholes_implemented, prepare_hedges, prepare_features
from clauses import add_cap_clause, add_knockout_clause, add_knockin_clause
from pfhedge.nn import Hedger, ExpectedShortfall, EntropicRiskMeasure, MultiLayerPerceptron, BlackScholes, WhalleyWilmott, Naked
from pfhedge.instruments import BrownianStock, HestonStock, LocalVolatilityStock, MertonJumpStock, RoughBergomiStock
from pfhedge.instruments import EuropeanOption, EuropeanBinaryOption, AmericanBinaryOption, LookbackOption, VarianceSwap, AsianOption, EuropeanForwardStartOption
import seaborn

def sigma_fn(time,spot):
    return 0.2*spot


if __name__ == "__main__":
    seaborn.set_style("whitegrid")
    n_qubits=2
    stock = HestonStock()
    derivative = EuropeanOption(stock,call=True)
    blackscholes = black_scholes_implemented(derivative)
    extra = EuropeanOption(stock)
    hedge = prepare_hedges(1e-4,stock)
    features = prepare_features(derivative,True)
    QUANTUM=False
    NTB = 0
    out = len(hedge)+NTB
    if QUANTUM:
        circuit = SimpleQuantumCircuit(n_qubits,4)
        model = MultiLayerHybrid(circuit.get_module(),n_qubits,n_qubits,n_layers=3,n_units=16, out_features=out)
    else:
        model = MultiLayerPerceptron(out_features=out)

    if NTB==1:
        model = NoTransactionBandNet(derivative,model)
    entropic = EntropicRiskMeasure(a=2)
    expected_shortfall = ExpectedShortfall()
    hedger = Hedger(model,inputs=features, criterion=expected_shortfall)
    if QUANTUM:
        history = hedger.fit(derivative,n_epochs=10,n_paths=250,n_times=1,hedge=hedge)
    else:
        history = hedger.fit(derivative,n_epochs=100,n_paths=5000,n_times=8,hedge=hedge)
    pnl = hedger.compute_pnl(derivative,n_paths=1000,hedge=hedge)
    save_pl_diagram(pnl, 'pldiagram.png')
    save_training_diagram(history,'trainingdiagram.png')
    print("Trained model:")
    print(expected_shortfall(pnl))
    if blackscholes:
        compmodel = WhalleyWilmott(derivative) 
        comphedger = Hedger(compmodel, inputs=compmodel.inputs())
        print("Whalley-Wilmott:")
        comp = comphedger.compute_pnl(derivative, n_paths=1000)
        save_pl_diagram(comp,'plbenchmark.png')
        print(expected_shortfall(comp))
    nohedger = Hedger(Naked(),inputs=["empty"])
    print("No hedge:")
    nohedge = nohedger.compute_pnl(derivative,n_paths=1000)
    print(expected_shortfall(nohedge))
    save_pl_diagram(nohedge,'plnone.png')