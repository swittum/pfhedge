import pennylane as qml
from abc import ABC,abstractmethod
class QuantumCircuit(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.n_inputs = 0
        self.n_outputs = 0
    @abstractmethod
    def get_module(self):
        raise NotImplementedError()

class SimpleQuantumCircuit(QuantumCircuit):
    def __init__(self, n_qubits=2, n_layers=4):
        super().__init__()
        dev = qml.device('default.qubit', wires=n_qubits)
        self.qnode = self._make_qnode(n_qubits,dev)
        self.weight_shapes = self._make_weight_shapes(n_qubits,n_layers)
        self.n_inputs = n_qubits
        self.n_outputs = n_qubits
    def _make_qnode(self, n_qubits, dev):
        @qml.qnode(dev)
        def qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
        return qnode
    def _make_weight_shapes(self, n_qubits, n_layers):
        return {"weights": (n_layers, n_qubits)}
    def get_module(self):
        return qml.qnn.TorchLayer(self.qnode,self.weight_shapes)



