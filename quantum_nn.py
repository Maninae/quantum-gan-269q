import pennylane as qml

from pennylane import numpy as np
from pennylane.optimize import GradientDescentOptimizer
from pennylane.ops import Hadamard, RX, CNOT, PauliX, PauliZ
from shor_code_model import real_shor_code
from pyquil.api import WavefunctionSimulator

from pennylane.ops import Hadamard, RX, CNOT

import random



##############################################################

test_dev = qml.device('forest.qvm', device='10q-pyqvm', noise=False, shots=1000)



def add_single_qubit_error(qbit_list):
    """
    Applies either a sign flip (Z-gate) or a bit flip (X-gate)
    or both to a randomly selected qubit to simulate a single
    bit error in a noisy channel.
    """
    qbit = random.randint(0, 8)
    if (random.random() < 0.5):
        qml.PauliX(qbit)
    if (random.random() < 0.5):
        qml.PauliZ(qbit)

def bit_encode(qbit1, qbit2, qbit3):
    qml.CNOT(wires=[qbit1, qbit2])
    qml.CNOT(wires=[qbit2, qbit3])


def hadamard_list(qbit_list):
    for qbit in qbit_list:
        qml.Hadamard(qbit)


# Declare quantum circuit
@qml.qnode(test_dev)
def shor_encoding(input_wire):

    first_level_qubits = (0, 3, 6)
    bit_encode(*first_level_qubits)

    # apply hadamard
    hadamard_list(first_level_qubits)

    for qubit in first_level_qubits:
        bit_encode(qubit, qubit+1, qubit+2)


    expectation_list = []

    # return qml.expval.PauliZ(wires=0)
    #
    for qbit in range(9):
        expecatation_val = qml.expval.PauliZ(wires=qbit)
        expectation_list.append(expecatation_val)

    print(expectation_list)

    return expectation_list


def shor_decoding_model(weights):
    # Our neural network
    for i in range(9):
        first = i
        second = 9
        qml.RX(weights[0], wires=first)
        qml.RY(weights[2], wires=first)
        qml.RZ(weights[4], wires=first)

        qml.CNOT(wires=[first, second])

        qml.RX(weights[1], wires=second)
        qml.RY(weights[3], wires=second)
        qml.RZ(weights[5], wires=second)


@qml.qnode(test_dev)
def shor_decoding_circuit(weights):
    # We need to wrap ground_truth_qbit_value in a bigger function, like the pennylane notebook does.
    # The API for pennylane doesn't let us pass in any more parameters
    # So we can then pass it into the decoding circuit

    # Run the model/circuit
    shor_decoding_model(weights)

    # Assume the decoding circuit has been run, and the output is on wire 9
    #  Create ground truth qubit on wire 0
    qml.reset(wires=0)  # This is PSEUDOCODE RN
    if ground_truth_qbit_value == 1:
        qml.X(0)

    #entangle and measure the parity
    qml.CNOT(0, 9)
    return qml.expval.PauliZ(wires=9)


def loss_function(weights):
    measurement = shor_decoding_circuit(weights)
    return -(measurement + 1) / 2


def train(model_function, weights):
    # A training loop. Use GDO?
    # Construct our CNOt loss
    alpha = 0.1
    optimizer = GradientDescentOptimizer(alpha)

    # Optimize D, fix G
    for it in range(50):
        disc_weights = optimizer.step(loss_function, weights)
        cost = optimizer(disc_weights)
        # if it % 1 == 0:
        print("Step {}: cost = {}".format(it + 1, cost))




##############################################################


if __name__ == "__main__":
    output = shor_encoding(0)
    print(output)

    # Other things if necessary