from typing import Optional,List
from math import ceil
import torch
from torch import Tensor

from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.doc import _set_docstring
from pfhedge._utils.str import _format_float
from pfhedge.nn.functional import european_payoff

from ..primary.base import BasePrimary
from .base import BaseDerivative


class MultiDerivative(BaseDerivative):
    r"""European option.

    The payoff of a European call option is given by:

    .. math::
        \mathrm{payoff} = \max(S - K, 0) ,

    where
    :math:`S` is the underlier's spot price at maturity and
    :math:`K` is the strike.

    The payoff of a European put option is given by:

    .. math::
        \mathrm{payoff} = \max(K - S, 0) .

    .. seealso::
        - :func:`pfhedge.nn.functional.european_payoff`

    Args:
        underlier (:class:`BasePrimary`): The underlying instrument of the option.
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.
        maturity (float, default=20/250): The maturity of the option.

    Attributes:
        dtype (torch.dtype): The dtype with which the simulated time-series are
            represented.
        device (torch.device): The device where the simulated time-series are.

    Examples:
        >>> import torch
        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanOption
        >>>
        >>> _ = torch.manual_seed(42)
        >>> derivative = EuropeanOption(BrownianStock(), maturity=5/250)
        >>> derivative.simulate(n_paths=2)
        >>> derivative.underlier.spot
        tensor([[1.0000, 1.0016, 1.0044, 1.0073, 0.9930, 0.9906],
                [1.0000, 0.9919, 0.9976, 1.0009, 1.0076, 1.0179]])
        >>> derivative.payoff()
        tensor([0.0000, 0.0179])

        Using custom ``dtype`` and ``device``.

        >>> derivative = EuropeanOption(BrownianStock())
        >>> derivative.to(dtype=torch.float64, device="cuda:0")
        EuropeanOption(
          ...
          (underlier): BrownianStock(..., dtype=torch.float64, device='cuda:0')
        )

        Make ``self`` a listed derivative.

        >>> from pfhedge.nn import BlackScholes
        >>>
        >>> pricer = lambda derivative: BlackScholes(derivative).price(
        ...     log_moneyness=derivative.log_moneyness(),
        ...     time_to_maturity=derivative.time_to_maturity(),
        ...     volatility=derivative.ul().volatility)
        >>> derivative = EuropeanOption(BrownianStock(), maturity=5/250)
        >>> derivative.list(pricer, cost=1e-4)
        >>> derivative.simulate(n_paths=2)
        >>> derivative.ul().spot
        tensor([[1.0000, 0.9788, 0.9665, 0.9782, 0.9947, 1.0049],
                [1.0000, 0.9905, 1.0075, 1.0162, 1.0119, 1.0220]])
        >>> derivative.spot
        tensor([[0.0113, 0.0028, 0.0006, 0.0009, 0.0028, 0.0049],
                [0.0113, 0.0060, 0.0130, 0.0180, 0.0131, 0.0220]])

        Add a knock-out clause with a barrier at 1.03:

        >>> _ = torch.manual_seed(42)
        >>> derivative = EuropeanOption(BrownianStock(), maturity=5/250)
        >>> derivative.simulate(n_paths=8)
        >>> derivative.payoff()
        tensor([0.0000, 0.0000, 0.0113, 0.0414, 0.0389, 0.0008, 0.0000, 0.0000])
        >>>
        >>> def knockout(derivative, payoff):
        ...     max = derivative.underlier.spot.max(-1).values
        ...     return payoff.where(max < 1.03, torch.zeros_like(max))
        >>>
        >>> derivative.add_clause("knockout", knockout)
        >>> derivative
        EuropeanOption(
          strike=1., maturity=0.0200
          clauses=['knockout']
          (underlier): BrownianStock(sigma=0.2000, dt=0.0040)
        )
        >>> derivative.payoff()
        tensor([0.0000, 0.0000, 0.0113, 0.0000, 0.0000, 0.0008, 0.0000, 0.0000])
    """

    def __init__(
        self,
        underlier: BasePrimary,
        derivatives: List[BaseDerivative],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.register_underlier("underlier", underlier)
        if len(derivatives) == 0:
            raise ValueError("No derivatives specified")
        self.maturity = max([der.maturity for der in derivatives])
        self.derivatives = derivatives
        #For compatibility with Whalley-Wilmott
        self.strike = 1.0
        self.steps = ceil(self.maturity / self.underlier.dt + 1)
        # TODO(simaki): Remove later. Deprecated for > v0.12.3
        if dtype is not None or device is not None:
            self.to(dtype=dtype, device=device)
            raise DeprecationWarning(
                "Specifying device and dtype when constructing a Derivative is deprecated."
                "Specify them in the constructor of the underlier instead."
            )

    def extra_repr(self) -> str:
        params = [der.extra_repr() for der in self.derivatives]
        return "; ".join(params)

    def payoff_fn(self) -> Tensor:
        return torch.sum(torch.stack([der.payoff() for der in self.derivatives]),0)
    
    def moneyness(self, time_step: Optional[int] = None, log: bool = False) -> Tensor:
        """Returns the moneyness of self.

        Moneyness reads :math:`S / K` where
        :math:`S` is the spot price of the underlying instrument and
        :math:`K` is the strike of the derivative.

        Args:
            time_step (int, optional): The time step to calculate
                the moneyness. If ``None`` (default), the moneyness is calculated
                at all time steps.
            log (bool, default=False): If ``True``, returns log moneyness.

        Shape:
            - Output: :math:`(N, T)` where
              :math:`N` is the number of paths and
              :math:`T` is the number of time steps.
              If ``time_step`` is given, the shape is :math:`(N, 1)`.

        Returns:
            torch.Tensor
        """
        k = len(self.derivatives)
        return torch.sum(torch.stack([der.moneyness(time_step,log) for der in self.derivatives]),dim=0)/k
    def time_to_maturity(self, time_step: Optional[int] = None) -> Tensor:
        """Returns the time to maturity of self.

        Args:
            time_step (int, optional): The time step to calculate
                the time to maturity. If ``None`` (default), the time to
                maturity is calculated at all time steps.

        Shape:
            - Output: :math:`(N, T)` where
              :math:`N` is the number of paths and
              :math:`T` is the number of time steps.
              If ``time_step`` is given, the shape is :math:`(N, 1)`.

        Returns:
            torch.Tensor
        """
        n_paths, n_steps = self.underlier.spot.size()
        if time_step is None:
            # Time passed from the beginning
            t = torch.arange(n_steps).to(self.underlier.spot) * self.underlier.dt
            return (t[-1] - t).unsqueeze(0).expand(n_paths, -1)
        else:
            time = n_steps - (time_step % n_steps) - 1
            t = torch.tensor([[time]]).to(self.underlier.spot) * self.underlier.dt
            return t.expand(n_paths, -1)
    