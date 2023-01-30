from typing import List
from pfhedge.instruments import BaseDerivative
from pfhedge.nn import BlackScholes
from .black_scholes import BlackScholesModuleFactory
import torch
from torch import Tensor
class BSMultiDerivative(torch.nn.Module):
    def __init__(self,derivatives: List[BaseDerivative],steps: int):
        super().__init__()
        self.BSmodules = [BlackScholes(der) for der in derivatives]
        self.steps = steps
        self.dt = derivatives[0].underlier.dt
    def forward(self, input: Tensor) -> Tensor:
        timestamp = self._get_timestamp(input[0][0][1].item())
        output = self.delta()
        return output[...,timestamp].unsqueeze(1).unsqueeze(1)
    def price(self)->Tensor:
        return torch.sum(torch.stack([der.price() for der in self.BSmodules]),0)
    def delta(self)->Tensor:
        return torch.sum(torch.stack([der.delta() for der in self.BSmodules]),0)
    def gamma(self, *args)->Tensor:
        output = torch.sum(torch.stack([der.gamma() for der in self.BSmodules]),0)
        if len(args) == 0:
            return output
        timestamp = self._get_timestamp(args[1][0][0].item())
        return output[...,timestamp].unsqueeze(1).unsqueeze(1)
    def vega(self)->Tensor:
        return torch.sum(torch.stack([der.vega() for der in self.BSmodules]),0)
    def theta(self)->Tensor:
        return torch.sum(torch.stack([der.theta() for der in self.BSmodules]),0)
    def inputs(self)->List[str]:
        return ['underlier_spot', 'time_to_maturity', 'volatility']
    def _get_timestamp(self,time_to_maturity: float):
        return self.steps-int(time_to_maturity/self.dt)-1
    @classmethod
    def from_derivative(cls,derivative):
        return cls(derivative.derivatives,derivative.steps)

factory = BlackScholesModuleFactory()
factory.register_module("MultiDerivative", BSMultiDerivative)
