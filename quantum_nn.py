import pennylane as qml

from pennylane import numpy as np
from pennylane.optimize import GradientDescentOptimizer
from pennylane.ops import Hadamard, RX, CNOT, PauliX, PauliZ
from shor_code_model import real_shor_code
from pyquil.api import WavefunctionSimulator

from pennylane.ops import Hadamard, RX, CNOT

import random



##############################################################

test_dev = qml.device('forest.qvm', device='12q-pyqvm', noise=False, shots=1000)
ground_truth = 0


def save_weights(weights, filename):
    np.save(filename, weights)

def load_weights(filename):
    return np.load(filename)


 # Compares the original shor encoding measurements with the ones from the learned circuit
def compare_ground_truth_and_circuit(num_iterations, weights):
    accumulated_error = 0.0

    reference_zero = get_reference_zero_qbit()

    print("Reference zero: %f" % reference_zero)

    for i in range(num_iterations):
        # Assume we're making sure we always get back a 0
        # First step is getting the corrupted Shor encdoing
        ground_truth = reset_circuit()[0]

        print("\nGROUND TRUTH :%f" % ground_truth)
        # now round ground truth
        # ground_truth = int(round(ground_truth))
        if random.random() < 0.5:
            print("FLIPPING BIT")
            ground_truth = flip_startbit()
            reset_circuit()
            ground_truth = (ground_truth + 1) % 2

        print("After rounding ground truth: %f" % ground_truth)


        recovered_bit_label = get_circuit_decoding_output(weights)

        accumulated_error += np.abs(ground_truth - recovered_bit_label)
        print("predicted label: %f" % recovered_bit_label)

        # reset
        reset_circuit()
        # reference_zero = get_reference_zero_qbit()

    print("Accumulated error: %f" % accumulated_error)

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
def reset_circuit(set_to_ones=False):

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

    qml.RX(0, wires=0)
    qml.RX(0, wires=1)
    qml.RX(0, wires=2)
    qml.RX(0, wires=3)
    qml.RX(0, wires=4)
    qml.RX(0, wires=5)
    qml.RX(0, wires=6)
    qml.RX(0, wires=7)
    qml.RX(0, wires=8)
    qml.RX(0, wires=9)

    if set_to_ones:
        print("Flipping a bit in the reset_circuit()")
        qml.PauliX(wires=0)

    # qml.expval.PauliZ(0)
    # qml.expval.PauliZ(1)
    # qml.expval.PauliZ(2)
    # qml.expval.PauliZ(3)
    # qml.expval.PauliZ(4)
    # qml.expval.PauliZ(5)
    # qml.expval.PauliZ(6)
    # qml.expval.PauliZ(7)
    # qml.expval.PauliZ(8)


    #TODO: Figure out whether we are supposed to measure in the Pauli-Z axis or not. Might have to be X-axis?
    return  qml.expval.PauliZ(0), qml.expval.PauliZ(1), qml.expval.PauliZ(2), qml.expval.PauliZ(3), qml.expval.PauliZ(4), qml.expval.PauliZ(5), qml.expval.PauliZ(6), qml.expval.PauliZ(7), qml.expval.PauliZ(8), qml.expval.PauliZ(9)



def correct_to_ground(ground_truth):
    if int(round(ground_truth)):
        qml.PauliX(wires=0)

@qml.qnode(test_dev)
def flip_startbit():
    print("flipping the bit")
    qml.PauliZ(wires=0)

    # Just return the value of the second qubit instead
    return qml.expval.PauliZ(wires=0)


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

        #TODO: validate that this circuit can interpret both the logical 0 and logical 1 case
        qml.RX(weights[i], wires=first)
        qml.RY(weights[i+1], wires=first)
        qml.RZ(weights[i+2], wires=first)

        qml.CNOT(wires=[first, output_wire])

        qml.RX(weights[i+3], wires=output_wire)
        qml.RY(weights[i+4], wires=output_wire)
        qml.RZ(weights[i+5], wires=output_wire)



@qml.qnode(test_dev)
def get_reference_zero_qbit():

    # qml.CNOT(wires=[11, 10])
    qml.RX(0.00, 11)

    #TODO: should this be a Pauli-Z rotation as well?
    return qml.expval.PauliZ(wires=11)


@qml.qnode(test_dev)
def test_zeroes():
    qml.PauliX(wires=0)
    qml.PauliX(wires=0)

    return qml.expval.PauliZ(wires=0)



@qml.qnode(test_dev)
def get_circuit_decoding_output(weights):
    # First apply the shor encoding to get the noisy 9 bits
    shor_encoding()
    shor_decoding_model(weights)

    # entangle and measure the parity
    # wire 10 should have a blank 0 qbit that hasn't been touched yet. CNOT with the output bit
    qml.CNOT(wires=[10, 9])
    return qml.expval.PauliZ(wires=9)



@qml.qnode(test_dev)
def shor_decoding_circuit(weights):

    # Flip so the qbit is set to ground truth
    # correct_to_ground(ground_truth)
    shor_encoding()


    # We need to wrap ground_truth_qbit_value in a bigger function, like the pennylane notebook does.
    # The API for pennylane doesn't let us pass in any more parameters
    # So we can then pass it into the decoding circuit

    # Run the model/circuit
    shor_decoding_model(weights)

    # Assume the decoding circuit has been run, and the output is on wire 9
    #  Create ground truth qubit on wire 0


    # TODO: To reset wire 0 just measure it to fix it
    # qml.expval.PauliZ(wires=10)

    # If the ground_truth bit is actually 1, then negate it so the later CNOT serves as a check that the output and original bit are the same?
    # if ground_truth == 1:
    #     qml.PauliX(9)

    #TODO: verify this correct to do
    #entangle and measure the parity
    # wire 10 should have a blank 0 qbit that hasn't been touched yet. CNOT with the output bit
    qml.CNOT(wires=[10, 9])
    return qml.expval.PauliZ(wires=9)



# @qml.qnode(test_dev)

def loss_function(weights):
    # Have shor_encoding prepare and also give back the true value of the bit

    # Get a blank slate for the working Shor bits


    ground_truth = reset_circuit()[0]
    print("GROUND TRUTH :%f" % ground_truth)
    # now round ground truth

    assert(int(round(ground_truth)) == -1.0)
    # ground_truth = int(round(ground_truth))


    # TODO: Ideally we'd like the qubit we are encoding to either 0 or 1. Problem is there are no if-statements in circuits smh
    # With random probability start with 1 as the ground truth bit
    if random.random() < 0.5:
        print("FLIPPING BIT")
        ground_truth = flip_startbit()

    print("After rounding ground truth: %f" % ground_truth)


    # print("entering shor decoding step")
    # if ground_truth == 1:
    #     print("FLIPPED")
    #     qml.PauliX(wires=0)

    measurement = shor_decoding_circuit(weights)

    reset_circuit()

    print("Recovered measurement:")
    print(measurement)

    #TODO: validate this is the right thing to do here. Another issue with training could just be a bad loss function
    print("loss value: %s" % str( (ground_truth - measurement) ** 2))
    return (ground_truth - measurement) ** 2


def train(weights):
    # A training loop. Use GDO?
    # Construct our CNOt loss
    alpha = 0.4
    optimizer = GradientDescentOptimizer(alpha)

    #TODO: Ideally, we want to train encodings of logical 0 and logical 1.
    # This has been tricky to figure out in terms of PennyLane QNode restrictions.
    # A possable work around is to to have two different optimizers. One for logical 0 and another for logical 1


    # Optimize D, fix G
    for it in range(1000):
        print("Iteration %d" % it)
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

    save_weights(weights, "circuit_weights")

    # Other things if necessary

    # test_flipp = test_zeroes()
    # print("tested X flip: %f" % test_flipp)


    weights = load_weights("circuit_weights.npy")
    compare_ground_truth_and_circuit(100, weights)