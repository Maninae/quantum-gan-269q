import pennylane as qml
from pennylane import numpy as np
# MUST import numpy from pennylane! So Pennylane can keep track of hybrid-quantum-classical backprop    


# For all devices, device() accepts the following arguments:
#   name: the name of the device to be loaded
#   wires: the number of subsystems to initialize the device with
dev = qml.device('default.qubit', wires=2)


def demo1():
    """ To use with Rigetti QPU's and QVM's:
          https://pennylane-forest.readthedocs.io/en/latest/usage.html#usage

        We need to install Pennylane-Forest, which allows us to interface 
        between different abstractions of quantum devices.

        forest.numpy_wavefunction: provides a PennyLane device for the pyQVM Numpy wavefunction simulator
        forest.wavefunction:       provides a PennyLane device for the Forest wavefunction simulator
        forest.qvm:                provides a PennyLane device for the Forest QVM and pyQuil pyQVM simulator
        forest.qpu:                provides a PennyLane device for Forest QPU hardware devices
    """
    dev_numpy = qml.device('forest.numpy_wavefunction', wires=2)
    dev_simulator = qml.device('forest.wavefunction', wires=2)
    dev_pyqvm = qml.device('forest.qvm', device='2q-pyqvm', shots=1000)
    dev_qvm = qml.device('forest.qvm', device='2q-qvm', shots=1000)
    dev_qpu = qml.device('forest.qpu', device='Aspen-0-12Q-A', shots=1000)

    # For QVM
    @qml.qnode(dev_qvm)
    def circuit(x, y, z):
        """ Really simple function. Rotate by x, y, z then return the value
        """
        qml.RZ(z, wires=[0])
        qml.RY(y, wires=[0])
        qml.RX(x, wires=[0])
        qml.CNOT(wires=[0, 1])
        return qml.expval.PauliX(wires=1)

    from pennylane_forest.ops import PSWAP

    # For a real, physical Rigetti QPU
    @qml.qnode(dev_qpu)
    def func(x, y):
        qml.BasisState(np.array([1, 1]), wires=0)
        qml.RY(x, wires=0)
        qml.RX(y, wires=1)
        PSWAP(0.432, wires=[0, 1])
        qml.CNOT(wires=[0, 1])
        return qml.expval.PauliZ(1)