import torch
class JaxLayer(torch.nn.Module):
    def __init__(qnode):
        self.qnode = qnode
    def forward(self,inputs):
        if len(inputs.shape) > 1:
            # If the input size is not 1-dimensional, unstack the input along its first dimension,
            # recursively call the forward pass on each of the yielded tensors, and then stack the
            # outputs back into the correct shape
            reconstructor = [self.forward(x) for x in torch.unbind(inputs)]
            return torch.stack(reconstructor)

        # If the input is 1-dimensional, calculate the forward pass as usual
        
        return self._evaluate_qnode(inputs)
    def _evaluate_qnode(self,inputs):
        return self.qnode(inputs)
    def backward(self,ctx,grad_output):
        pass