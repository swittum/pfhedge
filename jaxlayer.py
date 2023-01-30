import torch
import jax.numpy as jnp
import jax
import numpy as np
from torch.nn.parameter import Parameter
from jax2torch_func import j2t, t2j


def make_jax_function(qnode, grad):
    class JaxFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs, weights):
            ctx.save_for_backward(inputs, weights)
            return j2t(qnode(t2j(inputs), t2j(weights)))

        @staticmethod
        def backward(ctx, grad_output):
            (
                inputs,
                weights,
            ) = ctx.saved_tensors
            grads = grad(t2j(inputs), t2j(grad_output), t2j(weights))
            return j2t(grads[0]), j2t(grads[1])

    return JaxFunction


class JaxLayer(torch.nn.Module):
    def __init__(self, circuit):
        super().__init__()
        batched_func = jax.jit(
            jnp.vectorize(circuit.qnode, excluded=[1], signature="(n)->(m)")
        )
        self.weights = Parameter(torch.empty(circuit.weight_shape))
        torch.nn.init.uniform_(self.weights, a=0, b=2 * np.pi)

        def calc_grad(x, grad, weights):
            y, vjp = jax.vjp(circuit.qnode, x, weights)
            return vjp(grad)

        batched_calc_grad = jax.jit(
            jnp.vectorize(calc_grad, excluded=[2], signature="(n),(k)->(n),(l,n)")
        )
        self.func = make_jax_function(batched_func, batched_calc_grad)

    def forward(self, inputs):
        return self.func.apply(inputs, self.weights)
