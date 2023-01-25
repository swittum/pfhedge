import pennylane as qml
from abc import ABC
import numpy as np
import jax
class QuantumCircuit(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.n_inputs = 0
        self.n_outputs = 0
        self.qnodes = []

class SimpleQuantumCircuit(QuantumCircuit):
    def __init__(self, n_qubits=2, n_layers=4, n_measurements = 0):
        super().__init__()
        if n_measurements == 0:
            n_measurements = n_qubits
        if n_measurements > n_qubits or n_measurements < 0:
            raise ValueError("Invalid number of measurements")
        self.n_inputs = n_qubits
        self.n_outputs = n_measurements
        dev = qml.device('default.qubit.jax', wires=n_qubits)
        weights = self._get_weights((n_layers,n_qubits))
        #self.qnodes = [self._make_qnode(n_qubits,dev,weights,i) for i in range(n_measurements)]
        self.qnode = self._make_qnode(n_qubits,dev, weights,n_measurements)
    def _make_qnode(self, n_qubits, dev, weights, n_measurements):
        @jax.jit
        @qml.qnode(dev, interface='jax', shots=None, diff_method='best')
        def qnode(inputs):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_measurements)]
            #return qml.expval(qml.PauliZ(wires=i))
        return qnode
    def _get_weights(self, shape):
        return 2*np.pi*np.random.random_sample(shape)

