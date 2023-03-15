from typing import Optional
from typing import Union

import pytest
import torch
from torch.testing import assert_close

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import FloatingAsianOption

cls = FloatingAsianOption


class TestFloatingAsianOption:
    """
    pfhedge.instruments.FloatingAsianOption
    """

    @classmethod
    def setup_class(cls):
        torch.manual_seed(42)

    def test_payoff(self, device: Optional[Union[str, torch.device]] = "cpu"):
        derivative = FloatingAsianOption(BrownianStock(), mult=1.2).to(device)
        derivative.ul().register_buffer(
            "spot",
            torch.tensor([[1.0, 1.5, 3.5], [2.5, 3.0, 3.5], [4.0, 4.0, 1.0]]).to(
                device
            ),
        )
        # avg [2.0, 3.0, 3.0]
        result = derivative.payoff()
        expect = torch.tensor([1.1, 0.0, 0.0]).to(device)
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_payoff_gpu(self):
        self.test_payoff(device="cuda")

    def test_payoff_put(self, device: Optional[Union[str, torch.device]] = "cpu"):
        derivative = FloatingAsianOption(BrownianStock(), mult=0.8, call=False).to(device)
        derivative.ul().register_buffer(
            "spot",
            torch.tensor([[0.5, 3.0, 2.5], [3.5, 3.0, 2.5], [4.0, 4.0, 1.0]]).to(
                device
            ),
        )
        # avg [2.0, 3.0, 3.0]
        result = derivative.payoff()
        expect = torch.tensor([0.0, 0.0, 1.4]).to(device)
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_payoff_put_gpu(self):
        self.test_payoff_put(device="cuda")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        derivative = FloatingAsianOption(BrownianStock(dtype=dtype))
        derivative.simulate()
        assert derivative.payoff().dtype == dtype

        derivative = FloatingAsianOption(BrownianStock()).to(dtype=dtype)
        derivative.simulate()
        assert derivative.payoff().dtype == dtype

    def test_repr(self):
        derivative = FloatingAsianOption(BrownianStock(), maturity=1.0)
        expect = """\
FloatingAsianOption(
  mult=1., maturity=1.
  (underlier): BrownianStock(sigma=0.2000, dt=0.0040)
)"""
        assert repr(derivative) == expect

        derivative = FloatingAsianOption(BrownianStock(), call=False, maturity=1.0)
        expect = """\
FloatingAsianOption(
  call=False, mult=1., maturity=1.
  (underlier): BrownianStock(sigma=0.2000, dt=0.0040)
)"""
        assert repr(derivative) == expect

    def test_init_dtype_deprecated(self):
        with pytest.raises(DeprecationWarning):
            _ = cls(BrownianStock(), dtype=torch.float64)
