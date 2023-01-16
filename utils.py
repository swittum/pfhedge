from typing import List,Union
from pfhedge.instruments import BaseDerivative, EuropeanOption, EuropeanBinaryOption, AmericanBinaryOption, LookbackOption
from pfhedge.instruments import VarianceSwap, AsianOption, EuropeanForwardStartOption
from pfhedge.nn import BlackScholes
def prepare_hedges(transaction_costs: Union[float, List[float]],*args):
    num_hedges = len(args)
    if num_hedges==0:
        raise ValueError("No hedging instruments given")
    transaction_costs = (transaction_costs,) * num_hedges if isinstance(num_hedges, int) else num_hedges
    hedge = []
    if len(transaction_costs) != num_hedges:
        raise ValueError("Mismatched transaction cost list")
    for i,der in enumerate(args):
        if der.is_listed:
            der.cost = transaction_costs[i]
        else:
            list_derivative(der, transaction_costs[i])
        hedge.append(der)
    return hedge
def prepare_features(derivative: BaseDerivative, prev_hedge: bool):
    features = None
    if derivative.__class__ in (EuropeanOption,EuropeanBinaryOption,AmericanBinaryOption,LookbackOption):
        features = BlackScholes(derivative).inputs()
    if derivative.__class__ == AsianOption:
        if derivative.geom:
            features = ["log_moneyness", "mean_log_moneyness", "expiry_time", "volatility"]
        else:
            features = ["log_moneyness", "log_mean_moneyness", "expiry_time", "volatility"]
    if derivative.__class__ == EuropeanForwardStartOption:
        features = ["log_moneyness", "expiry_time", "volatility"]
    if derivative.__class__ == VarianceSwap:
        features= ['volatility', 'underlier_spot']
    if features is None:
        raise TypeError("Derivative unsupported")
    if prev_hedge:
        features.append("prev_hedge")
    return features
        


def list_derivative(derivative: BaseDerivative,cost: float):
    pricer = None
    if derivative.__class__ == AmericanBinaryOption or derivative.__class__ == LookbackOption:
        pricer = lambda derivative: BlackScholes(derivative).price(
        log_moneyness=derivative.log_moneyness(),
        time_to_maturity=derivative.time_to_maturity(),
        volatility=derivative.ul().volatility)
    if derivative.__class__ == AmericanBinaryOption or derivative.__class__ == LookbackOption:
        pricer = lambda derivative: BlackScholes(derivative).price(
        log_moneyness=derivative.log_moneyness(),
        max_log_moneyness = derivative.max_log_moneyness(),
        time_to_maturity=derivative.time_to_maturity(),
        volatility=derivative.ul().volatility)        
    if derivative.__class__ == VarianceSwap:
        pricer = lambda varswap: varswap.ul().variance - varswap.strike
    if pricer==None:
        raise TypeError("Unsupported derivative type, must have Black-Scholes support or be variance swap")
    derivative.list(pricer,cost)
    


        