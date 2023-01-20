import pennylane as qml
import jax
class SimpleQuantumCircuit:
    def __init__(self, n_qubits, n_layers):       
        dev = qml.device('default.qubit.jax', wires=n_qubits)
        self.qnode = self._make_qnode(n_qubits,dev)
        self.weight_shapes = self._make_weight_shapes(n_qubits,n_layers)
    def _make_qnode(self, n_qubits, dev):
        @qml.qnode(dev, interface='jax')
        def qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
        return qnode
    def _make_weight_shapes(self, n_qubits, n_layers):
        return {"weights": (n_layers, n_qubits)}
    def get_module(self):
        return qml.qnn.TorchLayer(self.qnode,self.weight_shapes)
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
    def get_module(self):
        return qml.qnn.TorchLayer(self.qnode,{})



