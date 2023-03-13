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