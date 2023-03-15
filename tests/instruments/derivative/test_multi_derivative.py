from typing import Union
from typing import Optional
import torch
import pytest
from torch.testing import assert_close
from pfhedge.instruments import BrownianStock,EuropeanOption,MultiDerivative

class TestMultiDerivative:
    """
    pfhedge.instruments.MultiDerivative
    """

    @classmethod
    def setup_class(cls):
        torch.manual_seed(42)
    def test_payoff(self, device: Optional[Union[str, torch.device]] = "cpu"):
        stock = BrownianStock()
        derivative1 = EuropeanOption(stock,strike=1.1).to(device)
        derivative2 = EuropeanOption(stock,call=False).to(device)
        mul = MultiDerivative(stock,[derivative1,derivative2])
        mul.simulate(5)
        result = mul.payoff()
        expect = derivative1.payoff()+derivative2.payoff()
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_payoff_gpu(self):
        self.test_payoff(device="cuda")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        stock = BrownianStock(dtype=dtype)
        derivative = MultiDerivative(stock,[EuropeanOption(stock)])
        derivative.simulate()
        assert derivative.payoff().dtype == dtype
        stock = BrownianStock().to(dtype=dtype)
        derivative = MultiDerivative(stock,[EuropeanOption(stock)])
        derivative.simulate()
        assert derivative.payoff().dtype == dtype

    def test_init_dtype_deprecated(self):
        stock = BrownianStock()
        with pytest.raises(DeprecationWarning):
            _ = MultiDerivative(stock,[EuropeanOption(stock)], dtype=torch.float64)