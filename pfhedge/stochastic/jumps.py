import torch
from torch import Tensor
from typing import Optional,Callable
def generate_jumps(n_paths: int,
    n_steps: int,
    jump_per_year=1.0,
    jump_mean=0.0,
    jump_std=0.3,
    dt: float = 1 / 250,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    engine: Callable[..., Tensor] = torch.randn,
) -> Tensor:
    r"""Returns time series containing a jump component following a Poisson distribution.

    The jump sizes follow a normal distribution.

    Args:
        n_paths (int): The number of simulated paths.
        n_steps (int): The number of time steps.
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
        >>> generate_merton_jump(2, 5, jump_per_year=30.0)
        tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.7933, 0.7933]])
    """
    poisson = torch.distributions.poisson.Poisson(rate=jump_per_year * dt)
    n_jumps = poisson.sample((n_paths, n_steps - 1)).to(dtype=dtype, device=device)

    jump_size = (
        jump_mean
        + engine(*(n_paths, n_steps - 1), dtype=dtype, device=device) * jump_std
    )
    jump = n_jumps * jump_size
    jump = torch.cat(
        [torch.zeros((n_paths, 1), dtype=dtype, device=device), jump], dim=1
    )
    return jump.cumsum(1)