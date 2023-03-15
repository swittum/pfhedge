from typing import Optional
from typing import Union

import pytest
import torch
from torch.testing import assert_close

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import AsianOption

cls = AsianOption


class TestAsianOption:
    """
    pfhedge.instruments.AsianOption
    """

    @classmethod
    def setup_class(cls):
        torch.manual_seed(42)

    def test_payoff(self, device: Optional[Union[str, torch.device]] = "cpu"):
        derivative = AsianOption(BrownianStock(), strike=2.0).to(device)
        derivative.ul().register_buffer(
            "spot",
            torch.tensor([[1.0, 1.0, 2.5], [2.5, 2.5, 2.5], [4.0, 4.0, 1.0]]).to(
                device
            ),
        )
        # avg [1.5, 2.5, 3.0]
        result = derivative.payoff()
        expect = torch.tensor([0.0, 0.5, 1.0]).to(device)
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_payoff_gpu(self):
        self.test_payoff(device="cuda")

    def test_payoff_put(self, device: Optional[Union[str, torch.device]] = "cpu"):
        derivative = AsianOption(BrownianStock(), strike=3.0, call=False).to(device)
        derivative.ul().register_buffer(
            "spot",
            torch.tensor([[5.0, 5.0, 2.0], [1.5, 1.0, 3.5], [1.5, 1.0, 2.0]]).to(
                device
            ),
        )
        # avg [4.0, 2.0, 1.5]
        result = derivative.payoff()
        expect = torch.tensor([0.0, 1.0, 1.5]).to(device)
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_payoff_put_gpu(self):
        self.test_payoff_put(device="cuda")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        derivative = AsianOption(BrownianStock(dtype=dtype))
        derivative.simulate()
        assert derivative.payoff().dtype == dtype

        derivative = AsianOption(BrownianStock()).to(dtype=dtype)
        derivative.simulate()
        assert derivative.payoff().dtype == dtype

    def test_repr(self):
        derivative = AsianOption(BrownianStock(), maturity=1.0)
        expect = """\
AsianOption(
  strike=1., maturity=1.
  (underlier): BrownianStock(sigma=0.2000, dt=0.0040)
)"""
        assert repr(derivative) == expect

        derivative = AsianOption(BrownianStock(), call=False, maturity=1.0)
        expect = """\
AsianOption(
  call=False, strike=1., maturity=1.
  (underlier): BrownianStock(sigma=0.2000, dt=0.0040)
)"""
        assert repr(derivative) == expect

    def test_init_dtype_deprecated(self):
        with pytest.raises(DeprecationWarning):
            _ = cls(BrownianStock(), dtype=torch.float64)
