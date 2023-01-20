import torch
import numpy as np
import jax
from jax2torch_func import jax2torch,j2t,t2j
def unstack_and_apply(inputs, func):
    shp = inputs.shape
    inputs = inputs.reshape(-1,shp[-1])
    #inputs = list(inputs)
    output = [j2t(func(t2j(x))) for x in inputs]
    return torch.stack(output).reshape(shp)
def tensor_to_array(tensor):
    #temp = np.array(tensor)
    return jax.numpy.asarray(tensor)
def array_to_tensor(array):
    #temp = np.array(array)
    return torch.tensor(array)
def make_jax_function(qnode,grad):
    class JaxFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inputs):
                ctx.save_for_backward(inputs)
                return unstack_and_apply(inputs,qnode)
            @staticmethod
            def backward(ctx, grad_output):
                input, =ctx.saved_tensors
                new_grad = unstack_and_apply(input, grad)
                return torch.mul(grad_output,new_grad)
    return JaxFunction

class JaxLayer(torch.nn.Module):
    def __init__(self,circuit):
        super().__init__()
        qnode = circuit.qnode
        grad = jax.jit(jax.grad(qnode))
        self.Func = make_jax_function(qnode,grad)
    def forward(self,inputs):
        return self.Func.apply(inputs)


class TorchJaxLayer(torch.nn.Module):
    def __init__(self,circuit):
        super().__init__()
        self.function = jax2torch(circuit.qnode)
    def forward(self,inputs):
        return unstack_and_apply(inputs,self.function)

class Torch2JaxLayer(torch.nn.Module):
    def __init__(self,circuit):
        super.__init__()
        self.function = circuit.qnode
        self.grad = jax.jit(jax.grad(self.function))