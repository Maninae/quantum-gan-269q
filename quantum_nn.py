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
ground_truth = 0



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

    #TODO: Ideally change this to reset all the qbits. Currently having issues with passing a parameter to this function
    # for i in range(9):
    #     qml.PauliX(wires=i)
    #
    # val = None
    # for i in reversed(range(9)):
    #     if i == 0:
    #         val = qml.expval.PauliX(i)
    #     else:
    #          qml.expval.PauliX(i)

    qml.PauliX(wires=0)
    return qml.expval.PauliX(0)


def correct_to_ground(ground_truth):
    if int(round(ground_truth)):
        qml.PauliX(wires=0)


def flip_startbit(qubit):
    print("flipping the bit")
    qml.PauliX(wires=qubit)



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
def shor_decoding_circuit(weights):

    # Flip so the qbit is set to ground truth
    # correct_to_ground(ground_truth)

    #TODO: Ideally we'd like the qubbit we are encoding to either 0 or 1
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

    # If the ground_truth bit is actually 1, then negate it so the later CNOT serves as a check that the output and original bit are teh same?
    if ground_truth == 1:
        qml.PauliX(9)

    #entangle and measure the parity
    # wire 10 should have a blank 0 qbit that hasn't been touched yet. CNOT with the output bit
    qml.CNOT(wires=[10, 9])
    return qml.expval.PauliZ(wires=9)



# @qml.qnode(test_dev)

def loss_function(weights):
    # Have shor_encoding prepare and also give back the true value of the bit

    # Get a blank slate for the working Shor bits
    ground_truth = reset_circuit()


    print("GROUND TRUTH :%f" % ground_truth)
    # now round ground truth
    ground_truth = int(round(ground_truth))
    print("AFter rounding ground truth: %f" % ground_truth)

    # print("entering shor decoding step")
    measurement = shor_decoding_circuit(weights)

    #TODO: validate this is the right thing to do here
    return -(measurement + 1) / 2


def train(weights):
    # A training loop. Use GDO?
    # Construct our CNOt loss
    alpha = 0.3
    optimizer = GradientDescentOptimizer(alpha)



    # Optimize D, fix G
    for it in range(200):
        weights = optimizer.step(loss_function, weights)
        cost = loss_function(weights)
        # if it % 1 == 0:
        print("Step {}: cost = {}".format(it + 1, cost))
        print("END STEP\n\n\n")

    return weights





##############################################################


if __name__ == "__main__":
    eps = 1e-2
    num_weights = 9 * 3 * 2
    weights = np.array([0.0] + [0] * (num_weights-1)) + np.random.normal(scale=eps, size=[num_weights])
    print("weights before")
    print(weights)
    print(weights.shape)

    before_weights = np.copy(weights)

    weights = train(weights)

    print("weights after")
    print(weights)

    abs_diff = np.sum(np.abs(before_weights - weights))
    print("Total end weight diff: %f" % abs_diff)

    # Other things if necessary