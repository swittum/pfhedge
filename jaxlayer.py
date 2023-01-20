import torch
import numpy as np
import jax
from jax2torch import jax2torch
def unstack_and_apply(inputs, func):
    shp = inputs.shape
    inputs = inputs.reshape(-1,shp[-1])
    inputs = list(inputs)
    output = [array_to_tensor(func(tensor_to_array(x))) for x in inputs]
    return torch.stack(output).reshape(shp)
def tensor_to_array(tensor):
    #temp = np.array(tensor)
    return jax.numpy.asarray(tensor)
def array_to_tensor(array):
    #temp = np.array(array)
    return torch.tensor(array)
class JaxFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs, qnode, grad):
            ctx.save_for_backward(inputs)
            ctx.qnode =qnode
            ctx.grad = grad
            return unstack_and_apply(inputs,qnode)
        @staticmethod
        def backward(ctx, grad_output):
            input, =ctx.saved_tensors
            new_grad = unstack_and_apply(input, ctx.grad)
            return torch.mul(grad_output,new_grad),None,None

class JaxLayer(torch.nn.Module):
    def __init__(self,circuit):
        super().__init__()
        self.qnode = circuit.qnode
        self.grad = jax.grad(self.qnode)
    def forward(self,inputs):
        return JaxFunction.apply(inputs,self.qnode,self.grad)


class TorchJaxLayer(torch.nn.Module):
    def __init__(self,circuit):
        super().__init__()
        self.function = jax2torch(circuit.qnode)
    def forward(self,inputs):
        return self.function(inputs)