import pennylane as qml

from pennylane import numpy as np
from pennylane.optimize import GradientDescentOptimizer
from pennylane.ops import Hadamard, RX, CNOT, PauliX, PauliZ
from shor_code_model import real_shor_code
from pyquil.api import WavefunctionSimulator

from pennylane.ops import Hadamard, RX, CNOT

import random



##############################################################

test_dev = qml.device('forest.qvm', device='11q-pyqvm', noise=False, shots=1000)



def add_single_qubit_error():
    """
    Applies either a sign flip (Z-gate) or a bit flip (X-gate)
    or both to a randomly selected qubit to simulate a single
    bit error in a noisy channel.
    """
    qbit = random.randint(0, 8)
    if random.random() < 0.3:
        qml.PauliX(qbit)
    elif random.random() < 0.6:
        # TODO: figure out why phase flip doesn't appear to be changing
        # the expectation values -- maybe because we are measuring
        # in the Z basis?
        # print(qbit, ": phase")
        # qml.Hadamard(qbit)
        qml.PauliZ(qbit)
    else:
        qml.PauliX(qbit)
        qml.PauliZ(qbit)

def bit_encode(qbit1, qbit2, qbit3):
    qml.CNOT(wires=[qbit1, qbit2])
    qml.CNOT(wires=[qbit2, qbit3])


def hadamard_list(qbit_list):
    for qbit in qbit_list:
        qml.Hadamard(qbit)

@qml.qnode(test_dev)
def reset_circuit():

    # for i in reversed(range(8,1)):
    #     qml.PauliX(wires=i)
    #
    # for i in reversed(range(8,1)):
    #     qml.expval.PauliZ(wires=i)

    qml.PauliX(wires=0)
    return qml.expval.PauliX(wires=0)


def correct_to_ground(ground_truth):
    if int(round(ground_truth)):
        qml.PauliX(wires=0)


def flip_startbit(qubit):
    print("flipping the bit")
    qml.PauliX(wires=qubit)



# Declare quantum circuit
# @qml.qnode(test_dev)
def shor_encoding():
    # Always have the 0 bit be the start bit
    # Measure it so it's set to 0

    first_level_qubits = (0, 3, 6)
    bit_encode(*first_level_qubits)

    # apply hadamard
    hadamard_list(first_level_qubits)

    for qubit in first_level_qubits:
        bit_encode(qubit, qubit+1, qubit+2)

    # Simulate single qubit errors
    add_single_qubit_error()


    # expectation_list = []
    # return qml.expval.PauliZ(wires=0)
    #
    # for qbit in range(9):
    #     expecatation_val = qml.expval.PauliZ(wires=qbit)
    #     expectation_list.append(expecatation_val)
    #
    # return expectation_list


def shor_decoding_model(weights):
    # Our neural network
    output_wire = 9
    for i in range(9):
        first = i
        qml.RX(weights[i], wires=first)
        qml.RY(weights[i+1], wires=first)
        qml.RZ(weights[i+2], wires=first)

        qml.CNOT(wires=[first, output_wire])

        qml.RX(weights[i+3], wires=output_wire)
        qml.RY(weights[i+4], wires=output_wire)
        qml.RZ(weights[i+5], wires=output_wire)


@qml.qnode(test_dev)
def shor_decoding_circuit(weights, ground_truth):

    # Flip so the qbit is set to ground truth
    # correct_to_ground(ground_truth)


    # With random probability start with 1 as the ground truth bit
    # if random.random() < 0.5:
    #     flip_startbit(0)
    #     ground_truth = 1




    shor_encoding()



    # We need to wrap ground_truth_qbit_value in a bigger function, like the pennylane notebook does.
    # The API for pennylane doesn't let us pass in any more parameters
    # So we can then pass it into the decoding circuit

    # Run the model/circuit
    shor_decoding_model(weights)

    # Assume the decoding circuit has been run, and the output is on wire 9
    #  Create ground truth qubit on wire 0

    # This is PSEUDOCODE RN
    # qml.reset(wires=0)


    # TODO: To reset wire 0 just measure it to fix it
    # qml.expval.PauliZ(wires=10)



    if ground_truth == 1:
        qml.X(10)

    #entangle and measure the parity
    # qml.CNOT(wires=[0, 9])
    return qml.expval.PauliZ(wires=9)




def loss_function(weights):
    # Have shor_encoding prepare and also give back the true value of the bit

    # Get a blank slate for the working Shor bits
    ground_truth = reset_circuit()


    print("GROUND TRUTH :%f" % ground_truth)
    # now round ground truth
    ground_truth = int(round(ground_truth))



    print("entering shor decoding step")
    measurement = shor_decoding_circuit(weights, ground_truth)

    return -(measurement + 1) / 2


def train(weights):
    # A training loop. Use GDO?
    # Construct our CNOt loss
    alpha = 0.6
    optimizer = GradientDescentOptimizer(alpha)



    # Optimize D, fix G
    for it in range(50):
        disc_weights = optimizer.step(loss_function, weights)
        cost = loss_function(disc_weights)
        # if it % 1 == 0:
        print("Step {}: cost = {}".format(it + 1, cost))
        print("END STEP\n\n\n")





##############################################################


if __name__ == "__main__":
    eps = 1e-2
    num_weights = 9 * 3 * 2
    weights = np.array([0.0] + [0] * (num_weights-1)) + np.random.normal(scale=eps, size=[num_weights])
    print("weights before")
    print(weights)

    before_weights = np.copy(weights)

    train(weights)

    print("weights after")
    print(weights)

    abs_diff = np.sum(np.abs(before_weights - weights))
    print("Total end weight diff: %f" % abs_diff)

    # Other things if necessary