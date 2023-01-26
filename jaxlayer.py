import torch
import jax.numpy as jnp
import jax
from jax2torch_func import j2t,t2j

def make_jax_function(qnode,grad):
    class JaxFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inputs):
                ctx.save_for_backward(inputs)
                return j2t(qnode(t2j(inputs)))
            @staticmethod
            def backward(ctx, grad_output):
                inputs, =ctx.saved_tensors
                return j2t(grad(t2j(inputs),t2j(grad_output))[0])
    return JaxFunction

class JaxLayer(torch.nn.Module):
    def __init__(self,circuit):
        super().__init__()
        batched_func = jax.jit(jnp.vectorize(circuit.qnode,signature='(n)->(m)'))
        def calc_grad(x,grad):
            y, vjp = jax.vjp(circuit.qnode,x)
            return vjp(grad)
        batched_calc_grad = jax.jit(jnp.vectorize(calc_grad,signature='(n),(k)->(n)'))
        self.func = make_jax_function(batched_func,batched_calc_grad)
    def forward(self,inputs):
        return self.func.apply(inputs)