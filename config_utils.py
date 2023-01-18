import warnings
from typing import List
from pfhedge.instruments import BaseInstrument, BasePrimary, BaseDerivative
from pfhedge.instruments import BrownianStock,HestonStock,MertonJumpStock,RoughBergomiStock
from pfhedge.instruments import EuropeanOption,EuropeanBinaryOption,LookbackOption
from utils import list_derivative
def dict_without_keys(dictionary: dict, *args: tuple[str]):
    copy = dict()
    for key in dictionary.keys():
        if not key in args:
            copy[key] = dictionary[key]
    return copy
def make_underlier(config: dict) -> BasePrimary:
    if not 'type' in config.keys():
        warnings.warn("No underlier type in configuration, using defaults.")
        underlier_type = BrownianStock
        cfg = {}
    else:
        options = {'BrownianStock':BrownianStock,
                   'HestonStock':HestonStock,
                   'MertonJumpStock':MertonJumpStock, 
                   'RoughBergomiStock':RoughBergomiStock}
        
        underlier_type = options[config['type']]
        cfg = dict_without_keys(config, 'type')
    return underlier_type(**cfg)
def make_derivative(config: dict, underlier: BasePrimary) -> BaseDerivative:
    if not 'type' in config.keys():
        warnings.warn("No derivative type in configuration, using defaults.")
        derivative_type = EuropeanOption
        cfg = {}
    else:
        options = {'EuropeanOption': EuropeanOption,
                   'EuropeanBinaryOption': EuropeanBinaryOption,
                   'LookbackOption': LookbackOption
                   }
        derivative_type = options[config['type']]
        cfg = dict_without_keys(config,'type','cost')
    return derivative_type(underlier=underlier,**cfg)
def make_hedge(config: dict, underlier:BasePrimary) -> List[BaseInstrument]:
    hedge = []
    if 'underlier' not in config.keys():
        warnings.warn("Not specified whether underlier is hedge, defaulting to True")
        hedge.append(underlier)
    elif config['underlier']:
        hedge.append(underlier)
    if 'derivatives' not in config.keys():
        derivative_list = []
    else:
        derivative_list = config['derivatives']
    for entry in derivative_list:
        der = make_derivative(entry, underlier)
        if 'cost' not in entry.keys():
            entry['cost'] = 0.0
        list_derivative(der, entry['cost'])
        hedge.append(der)
    return hedge




