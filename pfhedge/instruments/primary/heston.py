from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import Tensor

from pfhedge._utils.doc import set_attr_and_docstring
from pfhedge._utils.doc import set_docstring

from ...stochastic import generate_heston
from .base import Primary

TensorOrFloat = Union[Tensor, float]


class HestonStock(Primary):
    """A stock of which spot price and variance follow Heston process.

    See :func:`pfhedge.stochastic.generate_heston` for details of the process.

    Args:
        kappa (float, default=1.0): The parameter :math:`\\kappa`.
        theta (float, default=0.04): The parameter :math:`\\theta`.
        sigma (float, default=2.0): The parameter :math:`\\sigma`.
        rho (float, default=-0.7): The parameter :math:`\\rho`.
        cost (float, default=0.0): The transaction cost rate.
        dt (float, default=1/250): The intervals of the time steps.
        dtype (torch.device, optional): Desired device of returned tensor.
            Default: If None, uses a global default
            (see ``torch.set_default_tensor_type()``).
        device (torch.device, optional): Desired device of returned tensor.
            Default: if None, uses the current device for the default tensor type
            (see ``torch.set_default_tensor_type()``).
            ``device`` will be the CPU for CPU tensor types and
            the current CUDA device for CUDA tensor types.

    Buffers:
        - ``spot`` (``torch.Tensor``): The spot price of the instrument.
          This attribute is set by a method :func:`simulate()`.
          The shape is :math:`(N, T)` where
          :math:`N` is the number of simulated paths and
          :math:`T` is the number of time steps.
        - ``variance`` (``torch.Tensor``): The variance of the instrument.
          Note that this is different from the realized variance of the spot price.
          This attribute is set by a method :func:`simulate()`.
          The shape is :math:`(N, T)`.

    Examples:

        >>> from pfhedge.instruments import HestonStock
        >>>
        >>> _ = torch.manual_seed(42)
        >>> stock = HestonStock()
        >>> stock.simulate(n_paths=2, time_horizon=5/250)
        >>> stock.spot
        tensor([[1.0000, 0.9902, 0.9823, 0.9926, 0.9968, 1.0040],
                [1.0000, 0.9826, 0.9891, 0.9898, 0.9851, 0.9796]])
        >>> stock.variance
        tensor([[0.0400, 0.0408, 0.0411, 0.0417, 0.0422, 0.0393],
                [0.0400, 0.0457, 0.0440, 0.0451, 0.0458, 0.0472]])
        >>> stock.volatility
        tensor([[0.2000, 0.2020, 0.2027, 0.2041, 0.2054, 0.1982],
                [0.2000, 0.2138, 0.2097, 0.2124, 0.2140, 0.2172]])
    """

    spot: Tensor
    variance: Tensor

    def __init__(
        self,
        kappa: float = 1.0,
        theta: float = 0.04,
        sigma: float = 0.2,
        rho: float = -0.7,
        cost: float = 0.0,
        dt: float = 1 / 250,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.cost = cost
        self.dt = dt

        self.to(dtype=dtype, device=device)

    @property
    def default_init_state(self) -> Tuple[float, ...]:
        return (1.0, self.theta)

    @property
    def volatility(self) -> Tensor:
        """An alias for ``self.variance.sqrt()``."""
        return self.variance.clamp(min=0.0).sqrt()

    def simulate(
        self,
        n_paths: int = 1,
        time_horizon: float = 20 / 250,
        init_state: Optional[Tuple[TensorOrFloat, ...]] = None,
    ) -> None:
        """Simulate the spot price and add it as a buffer named ``spot``.

        The shape of the spot is :math:`(N, T)`, where :math:`N` is the number of
        simulated paths and :math:`T` is the number of time steps.
        The number of time steps is determinded from ``dt`` and ``time_horizon``.

        Args:
            n_paths (int, default=1): The number of paths to simulate.
            time_horizon (float, default=20/250): The period of time to simulate
                the price.
            init_state (tuple[torch.Tensor | float], default=(1.0,)): The initial
                state of the instrument.
                This is specified by ``(S0, V0)``, where ``S0`` and ``V0`` are
                the initial values of of spot and variance, respectively.
                If ``None`` (default), it uses the default value
                (See :func:`default_init_state`).
        """
        if init_state is None:
            init_state = self.default_init_state

        spot, variance = generate_heston(
            n_paths=n_paths,
            n_steps=int(time_horizon / self.dt + 1),
            init_state=init_state,
            kappa=self.kappa,
            theta=self.theta,
            sigma=self.sigma,
            rho=self.rho,
            dt=self.dt,
            dtype=self.dtype,
            device=self.device,
        )

        self.register_buffer("spot", spot)
        self.register_buffer("variance", variance)

    def __repr__(self) -> str:
        params = [
            f"kappa={self.kappa:.2e}",
            f"theta={self.theta:.2e}",
            f"sigma={self.sigma:.2e}",
            f"rho={self.rho:.2e}",
        ]
        if self.cost != 0.0:
            params.append(f"cost={self.cost:.2e}")
        params.append(f"dt={self.dt:.2e}")
        params += self.dinfo
        return self.__class__.__name__ + "(" + ", ".join(params) + ")"


# Assign docstrings so they appear in Sphinx documentation
set_docstring(HestonStock, "default_init_state", Primary.default_init_state)
set_attr_and_docstring(HestonStock, "to", Primary.to)