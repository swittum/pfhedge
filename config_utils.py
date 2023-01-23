import warnings
from typing import List
from torch.nn import Module
from pfhedge.nn import MultiLayerPerceptron, HedgeLoss, EntropicRiskMeasure, EntropicLoss, ExpectedShortfall
from pfhedge.instruments import BaseInstrument, BasePrimary, BaseDerivative
from pfhedge.instruments import BrownianStock,HestonStock,MertonJumpStock,RoughBergomiStock
from pfhedge.instruments import EuropeanOption,EuropeanBinaryOption,LookbackOption, AmericanBinaryOption, VarianceSwap, AsianOption, EuropeanForwardStartOption, FloatingAsianOption
from utils import list_derivative
from models import MultiLayerHybrid, NoTransactionBandNet
from quantum_circuits import QuantumCircuit,SimpleQuantumCircuit
from clauses import add_cap_clause, add_knockin_clause, add_knockout_clause
def dict_without_keys(dictionary: dict, *args: tuple[str]):
    copy = dict()
    for key in dictionary.keys():
        if not key in args:
            copy[key] = dictionary[key]
    return copy
def make_underlier(config: dict) -> BasePrimary:
    options = {'BrownianStock':BrownianStock,
                   'HestonStock':HestonStock,
                   'MertonJumpStock':MertonJumpStock, 
                   'RoughBergomiStock':RoughBergomiStock}
    underlier_type = options[config.get('type', 'BrownianStock')]
    cfg = dict_without_keys(config, 'type')
    return underlier_type(**cfg)
def add_clause(config: dict, derivative: BaseDerivative):
    options = {'cap': add_cap_clause,
               'knockout': add_knockout_clause,
               'knockin': add_knockin_clause
               }
    clause_type = options.get(config.get('type','none'), None)
    if clause_type == None:
        return
    cfg = dict_without_keys(config, 'type')
    clause_type(derivative, **cfg)
def make_derivative(config: dict, underlier: BasePrimary) -> BaseDerivative:
    options =  {'EuropeanOption': EuropeanOption,
                'EuropeanBinaryOption': EuropeanBinaryOption,
                'AmericanBinaryOption': AmericanBinaryOption,
                'LookbackOption': LookbackOption,
                'VarianceSwap': VarianceSwap,
                'AsianOption': AsianOption,
                'EuropeanForwardStartOption': EuropeanForwardStartOption,
                'FloatingAsianOption': FloatingAsianOption,
                }
    derivative_type = options[config.get('type', 'EuropeanOption')]
    cfg = dict_without_keys(config,'type','cost','clauses')
    derivative = derivative_type(underlier=underlier,**cfg)
    for clause in config.get('clauses', []):
        add_clause(clause, derivative)
    return derivative
def make_hedge(config: dict, underlier:BasePrimary) -> List[BaseInstrument]:
    hedge = []
    if config.get('underlier',True):
        hedge.append(underlier)
    derivative_list = config.get('derivatives',[])
    for entry in derivative_list:
        der = make_derivative(entry, underlier)
        list_derivative(der, entry.get('cost',0.0))
        hedge.append(der)
    return hedge
def make_circuit(config: dict) -> QuantumCircuit:
        options = {'SimpleQuantumCircuit': SimpleQuantumCircuit,
                   }
        model_type = options[config.get('type','SimpleQuantumCircuit')]
        cfg = dict_without_keys(config,'type')
        return model_type(**cfg)
def make_model(config: dict, n_hedges: int, derivative: BaseDerivative) -> Module:
    options = {'MultiLayerPerceptron': MultiLayerPerceptron,
                   'MultiLayerHybrid': MultiLayerHybrid,
                   }
    model_type = options[config.get('type','MultiLayerPerceptron')]
    NTB = config.get('NTB',True)
    if NTB:
        if n_hedges!= 1:
            raise ValueError("No Transaction Band not supported for more than one hedging instrument.")
        else:
            n_hedges = 2
    if model_type == MultiLayerPerceptron:       
        cfg = dict_without_keys(config,'type','NTB')
        model = model_type(out_features=n_hedges,**cfg)
    if model_type == MultiLayerHybrid:
        circuit_config = config.get('circuit',{})
        circuit = make_circuit(circuit_config)
        cfg = dict_without_keys(config,'type','NTB','circuit')
        model = MultiLayerHybrid(quantum=circuit,out_features=n_hedges, **cfg)
    if NTB:
        model = NoTransactionBandNet(derivative,model)
    return model
def make_criterion(config:dict)-> HedgeLoss:
    options = {'ExpectedShortfall':ExpectedShortfall,
                'EntropicRiskMeasure':EntropicRiskMeasure,
                'EntropicLoss':EntropicLoss,
                }
    loss_type = options[config.get('type', 'ExpectedShortfall')]
    cfg = dict_without_keys(config, 'type')
    return loss_type(**cfg)

        




