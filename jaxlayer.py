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
                #return unstack_and_apply(t2j(inputs),qnode)
            @staticmethod
            def backward(ctx, grad_output):
                inputs, =ctx.saved_tensors
                return j2t(grad(t2j(inputs),t2j(grad_output))[0])
                #return unstack_and_gradient(input, grad, grad_output)
                #return torch.mul(grad_output,new_grad)
    return JaxFunction

class JaxLayer(torch.nn.Module):
    def __init__(self,circuit):
        super().__init__()
        batched_func = jax.jit(jnp.vectorize(circuit.qnode,signature='(n)->(m)'))
        def calc_grad(x,grad):
            y, vjp = jax.vjp(circuit.qnode,x)
            return vjp(grad)
        batched_calc_grad = jax.jit(jnp.vectorize(calc_grad,signature='(n),(k)->(n)'))
        #qnode = self._make_function(circuit.qnodes)
        #grad_array = [jax.jit(jax.grad(x)) for x in circuit.qnodes]
        #grad_array = [jax.grad(x) for x in circuit.qnodes]
        #grad = self._make_function(grad_array)
        self.Func = make_jax_function(batched_func,batched_calc_grad)
    def _make_function(self,func_list):
        return lambda x: jax.numpy.asarray([func(x) for func in func_list])
    def forward(self,inputs):
        return self.Func.apply(inputs)