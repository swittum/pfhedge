from typing import List
from pfhedge.instruments import BaseDerivative
from pfhedge.nn import BlackScholes
from .black_scholes import BlackScholesModuleFactory
import torch
from torch import Tensor
class BSMultiDerivative:
    def __init__(self,derivatives: List[BaseDerivative]):
        self.BSmodules = [BlackScholes(der) for der in derivatives]
    def forward(self, input: Tensor) -> Tensor:
        return torch.sum(torch.stack([der.forward(input) for der in self.BSmodules]),0)
    def price(self)->Tensor:
        return torch.sum(torch.stack([der.price() for der in self.BSmodules]),0)
    def delta(self)->Tensor:
        return torch.sum(torch.stack([der.delta() for der in self.BSmodules]),0)
    def gamma(self)->Tensor:
        return torch.sum(torch.stack([der.gamma() for der in self.BSmodules]),0)
    def vega(self)->Tensor:
        return torch.sum(torch.stack([der.vega() for der in self.BSmodules]),0)
    def theta(self)->Tensor:
        return torch.sum(torch.stack([der.theta() for der in self.BSmodules]),0)
    @classmethod
    def from_derivative(cls,derivative):
        return cls(derivative.derivatives)

factory = BlackScholesModuleFactory()
factory.register_module("MultiDerivative", BSMultiDerivative)
