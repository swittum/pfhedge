import pennylane as qml
import numpy as np
import jax
class SimpleQuantumCircuit:
    def __init__(self, n_qubits, n_layers, n_measurements = 0):
        if n_measurements == 0:
            n_measurements = n_qubits
        dev = qml.device('default.qubit.jax', wires=n_qubits)
        weights = self._get_weights((n_layers,n_qubits))
        self.qnodes = [self._make_qnode(n_qubits,dev,weights,i) for i in range(n_measurements)]
        #self.qnode = self._make_qnode(n_qubits,dev, self._get_weights((n_layers,n_qubits)))
    def _make_qnode(self, n_qubits, dev, weights, i):
        @jax.jit
        @qml.qnode(dev, interface='jax', shots=None, diff_method='best')
        def qnode(inputs):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            #return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
            return qml.expval(qml.PauliZ(wires=i))
        return qnode
    def _get_weights(self, shape):
        return 2*np.pi*np.random.random_sample(shape)
class TestCircuit:
    def __init__(self):
        dev = qml.device("default.qubit.jax", wires=1, shots=None)
        @jax.jit
        @qml.qnode(dev, interface='jax', shots=None, diff_method='best')
        def circuit(inputs):
            qml.RX(inputs[1], wires=0)
            qml.Rot(inputs[0], inputs[1], inputs[2], wires=0)
            return qml.expval(qml.PauliZ(0))
        self.qnode=circuit
        self.qnodes = [self.qnode]



