import pennylane as qml

from pennylane import numpy as np
from pennylane.optimize import GradientDescentOptimizer, AdamOptimizer
from pennylane.ops import Hadamard, RX, CNOT, PauliX, PauliZ
from pyquil.api import WavefunctionSimulator

from pennylane.ops import Hadamard, RX, CNOT
from pennylane_forest.ops import CCNOT

import random
import pickle
import matplotlib.pyplot as plt



##############################################################

test_dev = qml.device('forest.qvm', device='12q-pyqvm', noise=False, shots=1000)
ground_truth = 0

recorded_loss_list = []
recorded_measurement_list = []


def pickle_list(list_obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(list_obj, f)

def plot_loss(filename):
    # Load the pickle file
    loss_values = None
    with open(filename, 'rb') as pickle_file:
        loss_values = pickle.load(pickle_file)

    # Matpolot lib it
    plt.plot(loss_values)
    plt.xlabel("Iteration #")
    plt.ylabel("MSE Loss")
    plt.title("Shor-inspired Model Loss with lr=0.01")
    plt.savefig(filename[:-4] + ".png")
    plt.show()


def save_weights(weights, filename):
    np.save(filename, weights)


def load_weights(filename):
    # DEPRECATED
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
            # ground_truth = (ground_truth + 1) % 2
            ground_truth = (-1) * ground_truth

        print("After rounding ground truth: %f" % ground_truth)


        recovered_bit_label = get_circuit_decoding_output(weights)

        error = np.abs(ground_truth - recovered_bit_label)
        accumulated_error += error
        print("predicted label: %f" % recovered_bit_label)
        print("added to error: %f" % error)

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
    p = random.random()
    if p < 0.33:
        qml.PauliX(qbit)
    elif p < 0.66:
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
    """
    Used in loss_function, so this doesn't have to be a qnode.
    """
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

    #TODO: Figure out whether we are supposed to measure in the Pauli-Z axis or not. Might have to be X-axis?
    return qml.expval.PauliZ(0), qml.expval.PauliZ(1), qml.expval.PauliZ(2), qml.expval.PauliZ(3), qml.expval.PauliZ(4), qml.expval.PauliZ(5), qml.expval.PauliZ(6), qml.expval.PauliZ(7), qml.expval.PauliZ(8), qml.expval.PauliZ(9)



def correct_to_ground(ground_truth):
    if int(round(ground_truth)):
        qml.PauliX(wires=0)

@qml.qnode(test_dev)
def flip_startbit():
    print("flipping the bit")
    qml.PauliX(wires=0)
    qml.PauliX(wires=1)

    # Just return the value of the second qubit instead
    return qml.expval.PauliZ(wires=1)


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


def shor_decoding_model_charissa_attempt(weights):
    assert (weights.shape == (2,))

    for i in [0, 3, 6]:
        qml.CNOT(wires=[i, i+1])
        qml.CNOT(wires=[i, i+2])
        CCNOT(wires=[i+1, i+2, i])

        # In the traditional Shor decoding circuit, there is an H gate here.
        # See if these rotations can be learned to imitate a Hadamard gate
        qml.RX(weights[0], wires=i)
        qml.RZ(weights[1], wires=i)

    qml.CNOT(wires=[0, 3])
    qml.CNOT(wires=[0, 6])
    CCNOT(wires=[3, 6, 0])

    # Output the qubit to wire 9
    output_wire = 9
    qml.CNOT(wires=[0, output_wire])
    qml.CNOT(wires=[3, output_wire])
    qml.CNOT(wires=[6, output_wire])



def shor_decoding_model_traditional(weights):
    assert(weights.shape == (3,))

    for i in [0, 3, 6]:
        qml.CNOT(wires=[i, i+1])
        qml.CNOT(wires=[i, i+2])
        CCNOT(wires=[i+1, i+2, i])

        # In the traditional Sho decoding circuit, there is an H gate here.
        # See if these rotations can be learned to imitate a Hadamard gate
        qml.RZ(weights[0], wires=i)
        qml.RX(weights[1], wires=i)
        qml.RZ(weights[2], wires=i)

    qml.CNOT(wires=[0, 3])
    qml.CNOT(wires=[0, 6])
    CCNOT(wires=[3, 6, 0])

    # Output the qubit to wire 9
    output_wire = 9
    qml.CNOT(wires=[0, output_wire])




def shor_decoding_model(weights):
    nb_layers, nb_qubits, _ = weights.shape
    assert weights.shape[2] == 2
    
    output_wire = 9
    assert nb_qubits == output_wire
    
    # Our neural network: the ansatz from class
    for i in range(nb_layers):

        for j in range(nb_qubits):
            # For wires 0-8 inclusive
            qml.RX(weights[i, j, 0], wires=j)
            # qml.RY(weights[i, j, 1], wires=j)
            qml.RZ(weights[i, j, 1], wires=j)

        for j in range(nb_qubits - 1):
            qml.CNOT(wires=[j, j+1])


    qml.CNOT(wires=[8, output_wire])



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
def get_circuit_decoding_output(weights, model_fn=shor_decoding_model):
    """
    Used only in compare_ground_truth_and_circuit.
    """
    # First apply the shor encoding to get the noisy 9 bits
    shor_encoding()
    model_fn(weights)

    # entangle and measure the parity
    # wire 10 should have a blank 0 qbit that hasn't been touched yet. CNOT with the output bit
    qml.CNOT(wires=[10, 9])
    return qml.expval.PauliZ(wires=9)



@qml.qnode(test_dev)
def shor_decoding_circuit(weights, model_fn):
    """
    Purely used inside the loss_function.
    """
    # Run the model/circuit
    shor_encoding()
    model_fn(weights)
    return qml.expval.PauliZ(wires=9)
    



def loss_function_MSE(weights):
    return loss_function(weights, as_probability=False)

def loss_function_probability(weights):
    return loss_function(weights, as_probability=True)



@qml.qnode(test_dev)
def encode_and_decode():
    shor_encoding()
    # shor_decoding_model(weights)  # Tried to refactor, but bug upon passing in model as kwarg
    shor_decoding_model_traditional(weights)
    measurement = qml.expval.PauliZ(wires=9)
    return measurement

def loss_function(weights, as_probability=False):
    # Have shor_encoding prepare and also give back the true value of the bit
    # Get a blank slate for the working Shor bits


    ground_truth = reset_circuit()[0]
    print("GROUND TRUTH :{}".format(ground_truth))
    assert(int(round(ground_truth)) == -1.0)

    # TODO: Ideally we'd like the qubit we are encoding to either 0 or 1. Problem is there are no if-statements in circuits smh
    # With random probability start with 1 as the ground truth bit
    if random.random() < 0.5:
        print("FLIPPING BIT")
        flip_startbit()
        ground_truth = 1.0
    print("After [maybe] flipping, ground truth: %f" % ground_truth)

    measurement = encode_and_decode()

    reset_circuit()

    print("Recovered measurement:")
    print(measurement)

    #TODO: validate this is the right thing to do here. Another issue with training could just be a bad loss function
    if as_probability:
        # Small epsiolon for numerical stabilty 
        epsilon = 0.000001
        p = (measurement + 1) / 2
        loss = -np.log(p + epsilon) if (ground_truth == 1) else -np.log(1. - p + epsilon)
    else:
        loss = (ground_truth - measurement) ** 2

    if isinstance(loss, float):
        recorded_loss_list.append(loss)
        recorded_measurement_list.append(measurement)
    else:
        recorded_loss_list.append(loss._value)
        recorded_measurement_list.append(measurement._value)

    print("loss value: %s" % str(loss))
    return loss


def train(weights, loss_fn, pickle_suffix=""):
    # A training loop. Use GDO?
    # Construct our CNOt loss
    alpha = 0.01
    optimizer = AdamOptimizer(alpha)

    #TODO: Ideally, we want to train encodings of logical 0 and logical 1.
    # This has been tricky to figure out in terms of PennyLane QNode restrictions.
    # A possable work around is to to have two different optimizers. One for logical 0 and another for logical 1

    # Optimize D, fix G
    for it in range(15000):
        print("Iteration %d" % it)
        print("Step {}".format(it + 1))
        weights = optimizer.step(loss_fn, weights)
        print("END STEP\n\n\n")


    # Save our recorded loss values
    pickle_list(recorded_loss_list, "logs/recorded_loss_list_%s.pkl" % pickle_suffix)
    pickle_list(recorded_measurement_list, "logs/recorded_measurment_list_%s.pkl" % pickle_suffix)
    return weights


##############################################################


if __name__ == "__main__":
    
    ##################################################
    #      Training 2 parameters to learn H gate     #
    #         shor_decoding_model_traditional()      #
    ##################################################
    weights = (np.pi / 3) * np.random.randn(3)
    print(type(weights))
    
    weights = train(weights, 
                    loss_function_MSE,
                    pickle_suffix="0605_1408_traditional")
    

    plot_loss("logs/recorded_loss_list_0605_1408_traditional.pkl")


    # abs_diff = np.sum(np.abs(before_weights - weights))
    # print("Total end weight diff: %f" % abs_diff)
    #
    # save_weights(weights, "circuit_weights")
    #
    print("weights before")
    print(weights)
    print(weights.shape)

    before_weights = np.copy(weights)

    # weights = (np.pi / 3) * np.random.randn(3, 2)
    weights = train(weights, loss_function_probability)

    print("weights after")
    print(weights)

    abs_diff = np.sum(np.abs(before_weights - weights))
    print("Total end weight diff: %f" % abs_diff)

    save_weights(weights, "circuit_weights")

    #
    # # Other things if necessary
    #
    #
    # # nb_layers = 5
    # # nb_qubits = 9
    # # weights = 0.1 * np.random.randn(nb_layers, nb_qubits, 2)
    # weights = np.load("circuit_weights.npy")
    compare_ground_truth_and_circuit(100, weights)
