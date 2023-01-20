import torch
import numpy as np
import jax
from jax2torch_func import jax2torch,j2t,t2j
def unstack_and_apply(inputs, func):
    shp = inputs.shape
    inputs = inputs.reshape(-1,shp[-1])
    #inputs = list(inputs)
    output = [j2t(func(t2j(x))) for x in inputs]
    output = torch.stack(output)
    return output.reshape((shp[0],shp[1],-1))
def unstack_and_gradient(inputs, grad_func, grad):
    shp = inputs.shape
    inputs = inputs.reshape(-1,shp[-1])
    grad = grad.reshape(-1,grad.shape[-1])
    #inputs = list(inputs)
    output = [torch.matmul(grad[i,...], j2t(grad_func(t2j(inputs[i,...])))) for i in range(inputs.shape[0])]
    #output = [j2t(grad_func(t2j(x))) for x in inputs]
    output = torch.stack(output)
    #output = torch.bmm(output,grad)
    return output.reshape((shp[0],shp[1],-1))
def make_jax_function(qnode,grad):
    class JaxFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inputs):
                ctx.save_for_backward(inputs)
                return unstack_and_apply(inputs,qnode)
            @staticmethod
            def backward(ctx, grad_output):
                input, =ctx.saved_tensors
                return unstack_and_gradient(input, grad, grad_output)
                #return torch.mul(grad_output,new_grad)
    return JaxFunction

class JaxLayer(torch.nn.Module):
    def __init__(self,circuit):
        super().__init__()
        qnode = self._make_function(circuit.qnodes)
        grad_array = [jax.jit(jax.grad(x)) for x in circuit.qnodes]
        grad = self._make_function(grad_array)
        self.Func = make_jax_function(qnode,grad)
    def _make_function(self,func_list):
        return lambda x: jax.numpy.asarray([func(x) for func in func_list])
    def forward(self,inputs):
        return self.Func.apply(inputs)