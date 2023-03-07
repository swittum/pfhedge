from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import Tensor
from math import pi

from pfhedge._utils.typing import TensorOrScalar
from pfhedge.stochastic._utils import cast_state

def generate_variance_gamma(
    n_paths: int,
    n_steps: int,
    init_state: Union[Tuple[TensorOrScalar, ...], TensorOrScalar] = (1.0,),
    sigma: float = 1.0,
    theta: float = 0.0,
    kappa: float = 1.0,
    dt: float = 1 / 250,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    engine: Callable[..., Tensor] = torch.randn,
) -> Tensor:
    r"""Returns time series following the Merton's Jump Diffusion Model.

    The time evolution of the process is given by:

    .. math::

        \frac{dS(t)}{S(t)} = (\mu - \lambda k) dt + \sigma dW(t) + dJ(t) .

    Reference:
     - Merton, R.C., Option pricing when underlying stock returns are discontinuous,
       Journal of Financial Economics, 3 (1976), 125-144.
     - Gugole, N. (2016). Merton jump-diffusion model versus the black and scholes approach for the log-returns and volatility
       smile fitting. International Journal of Pure and Applied Mathematics, 109(3), 719-736.

    Args:
        n_paths (int): The number of simulated paths.
        n_steps (int): The number of time steps.
        init_state (tuple[torch.Tensor | float], default=(1.0,)): The initial state of
            the time series.
            This is specified by a tuple :math:`(S(0),)`.
            It also accepts a :class:`torch.Tensor` or a :class:`float`.
        mu (float, default=0.0): The parameter :math:`\mu`,
            which stands for the dirft coefficient of the time series.
        sigma (float, default=0.2): The parameter :math:`\sigma`,
            which stands for the volatility of the time series.
        jump_per_year (float, default=1.0): The frequency of jumps in one year.
        jump_mean (float, default=0.0): The mean of jumnp sizes.
        jump_std (float, default=0.3): The deviation of jump sizes.
        dt (float, default=1/250): The intervals of the time steps.
        dtype (torch.dtype, optional): The desired data type of returned tensor.
            Default: If ``None``, uses a global default
            (see :func:`torch.set_default_tensor_type()`).
        device (torch.device, optional): The desired device of returned tensor.
            Default: If ``None``, uses the current device for the default tensor type
            (see :func:`torch.set_default_tensor_type()`).
            ``device`` will be the CPU for CPU tensor types and the current CUDA device
            for CUDA tensor types.
        engine (callable, default=torch.randn): The desired generator of random numbers
            from a standard normal distribution.
            A function call ``engine(size, dtype=None, device=None)``
            should return a tensor filled with random numbers
            from a standard normal distribution.

    Shape:
        - Output: :math:`(N, T)` where
          :math:`N` is the number of paths and
          :math:`T` is the number of time steps.

    Returns:
        torch.Tensor

    Examples:
        >>> from pfhedge.stochastic import generate_merton_jump
        >>>
        >>> _ = torch.manual_seed(42)
        >>> generate_merton_jump(2, 5)
        tensor([[1.0000, 0.9905, 1.0075, 1.0161, 1.0118],
                [1.0000, 1.0035, 1.0041, 1.0377, 1.0345]])
    """
    init_state = cast_state(init_state, dtype=dtype, device=device)
    gamma = torch.distributions.gamma.Gamma(dt/kappa,1.0)
    S = kappa * gamma.sample((n_paths,n_steps))
    normal = torch.distributions.normal.Normal(0,1)
    N = normal.sample((n_paths,n_steps,))
    X = sigma*N*S.sqrt()+theta*S
    X[:,0] = 0.0
    return init_state[0]*S.cumsum(1).exp()