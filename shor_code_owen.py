from typing import List
import numpy as np

from pyquil import Program
from pyquil.gates import MEASURE, I, CNOT, H, X, Z
from pyquil.quil import address_qubits
from pyquil.quilatom import QubitPlaceholder
from pyquil.api import QVMConnection

##
############# YOU MUST COMMENT IN THESE TWO LINES FOR IT TO WORK WITH THE AUTOGRADER
import subprocess
subprocess.Popen("/src/qvm/qvm -S > qvm.log 2>&1", shell=True)


# Do not change this SEED value you or your autograder score will be incorrect.
qvm = QVMConnection(random_seed=1337)


def bit_flip_channel(prob: float):
    noisy_I = np.sqrt(1-prob) * np.asarray([[1, 0], [0, 1]])
    noisy_X = np.sqrt(prob) * np.asarray([[0, 1], [1, 0]])
    return [noisy_I, noisy_X]


def phase_flip_channel(prob: float):
    # Noisy I, Z
    noisy_I = np.sqrt(1-prob) * np.asarray([[1, 0], [0, 1]])
    noisy_Z = np.sqrt(prob) * np.asarray([[1, 0], [0, -1]])
    return [noisy_I, noisy_Z]

def depolarizing_channel(prob: float):
    # Noisy I, X, Y, Z
    noisy_I = np.sqrt(1-prob) * np.asarray([[1, 0],    [0, 1]])
    noisy_X = np.sqrt(prob/3) * np.asarray([[0, 1],    [1, 0]])
    noisy_Y = np.sqrt(prob/3) * np.asarray([[0, -1.j], [1.j, 0]])
    noisy_Z = np.sqrt(prob/3) * np.asarray([[1, 0],    [0, -1]])
    return [noisy_I, noisy_X, noisy_Y, noisy_Z]


#################   Helper for readout names   #######################
nonce = 0
def get_unique_readout_name():
    global nonce
    nonce += 1
    return "ro_%06d" % nonce
######################################################################


def bit_code(qubit: QubitPlaceholder, noise=None) -> (Program, List[QubitPlaceholder]):

    ### Do your encoding step here
    x1 = QubitPlaceholder()
    x2 = QubitPlaceholder()
    code_register = [qubit, x1, x2]  # the List[QubitPlaceholder] of the qubits you have encoded into
    
    pq = Program()  # the Program that does the encoding
    new_readout_name = get_unique_readout_name()
    ro = pq.declare(new_readout_name, "BIT", 2)

    # Encoding
    pq += CNOT(qubit, x1)
    pq += CNOT(qubit, x2)

    ##########################################################################
    # DON'T CHANGE THIS CODE BLOCK. It applies the errors for simulations
    if noise is None:
        pq += [I(qq) for qq in code_register]
    else:
        pq += noise(code_register)
    ##########################################################################

    ### Do your decoding and correction steps here
    
    a1 = QubitPlaceholder()  # ancilla measuring difference between qubit, x1
    pq += CNOT(qubit, a1)
    pq += CNOT(x1, a1)
    
    a2 = QubitPlaceholder()  # ancilla measuring difference between qubit, x2
    pq += CNOT(qubit, a2)
    pq += CNOT(x2, a2)

    pq += MEASURE(a1, ro[0])
    pq += MEASURE(a2, ro[1])

    # Error correction: apply X gate to different qubits depending on the readout measurements
    ancillas_equal_00_branch = Program()           # Fix nothing
    ancillas_equal_01_branch = Program(X(x2))      # Fix x2
    a1_equals_0_branch = Program().if_then(ro[1], ancillas_equal_01_branch, ancillas_equal_00_branch)
    
    ancillas_equal_10_branch = Program(X(x1))      # Fix x1
    ancillas_equal_11_branch = Program(X(qubit))   # Fix qubit
    a1_equals_1_branch = Program().if_then(ro[1], ancillas_equal_11_branch, ancillas_equal_10_branch)

    pq.if_then(ro[0], a1_equals_1_branch, a1_equals_0_branch)  # Hope this if then is right? 1=True?

    # Instantiate the placeholders with actual qubits
    address_qubits(pq)    

    return pq, code_register


def phase_code(qubit: QubitPlaceholder, noise=None) -> (Program, List[QubitPlaceholder]):

    ### Do your encoding step here
    x1 = QubitPlaceholder()
    x2 = QubitPlaceholder()
    code_register = [qubit, x1, x2]  # the List[QubitPlaceholder] of the qubits you have encoded into
    
    pq = Program()  # the Program that does the encoding
    new_readout_name = get_unique_readout_name()
    ro = pq.declare(new_readout_name, "BIT", 2)

    # Encoding
    pq += CNOT(qubit, x1)
    pq += CNOT(qubit, x2)
    
    # Apply the three Hadamard gates
    pq += H(qubit)
    pq += H(x1)
    pq += H(x2)
    ##########################################################################
    # DON'T CHANGE THIS CODE BLOCK. It applies the errors for simulations
    if noise is None:
        pq += [I(qq) for qq in code_register]
    else:
        pq += noise(code_register)
    ##########################################################################

    ### Do your decoding and correction steps here
    
    # Unapply (i.e. reapply) the three Hadamard gates
    pq += H(qubit)
    pq += H(x1)
    pq += H(x2)

    a1 = QubitPlaceholder()  # ancilla measuring difference between qubit, x1
    pq += CNOT(qubit, a1)
    pq += CNOT(x1, a1)
    
    a2 = QubitPlaceholder()  # ancilla measuring difference between qubit, x2
    pq += CNOT(qubit, a2)
    pq += CNOT(x2, a2)

    pq += MEASURE(a1, ro[0])
    pq += MEASURE(a2, ro[1])

    # Error correction: apply X gate to different qubits depending on the readout measurements
    ancillas_equal_00_branch = Program()           # Fix nothing
    ancillas_equal_01_branch = Program(X(x2))      # Fix x2
    a1_equals_0_branch = Program().if_then(ro[1], ancillas_equal_01_branch, ancillas_equal_00_branch)
    
    ancillas_equal_10_branch = Program(X(x1))      # Fix x1
    ancillas_equal_11_branch = Program(X(qubit))   # Fix qubit
    a1_equals_1_branch = Program().if_then(ro[1], ancillas_equal_11_branch, ancillas_equal_10_branch)

    pq.if_then(ro[0], a1_equals_1_branch, a1_equals_0_branch)

    # Instantiate the placeholders with actual qubits
    address_qubits(pq)    

    return pq, code_register


def shor(qubit: QubitPlaceholder, noise=None) -> (Program, List[QubitPlaceholder]):
    # Note that in order for this code to work properly, you must build your Shor code using the phase code and
    # bit code methods above

    pq = Program()
    code_register = []

    pq_phase, code_register_phase = phase_code(qubit, noise=noise)
    pq += pq_phase

    for output_qubit in code_register_phase:
        pq_bit, code_register_bit = bit_code(output_qubit, noise=noise)
        pq += pq_bit
        code_register.extend(code_register_bit)

    return pq, code_register



def run_code(error_code, noise, trials=10):
    """ Takes in an error_code function (e.g. bit_code, phase_code or shor) and runs this code on the QVM"""
    pq, code_register = error_code(QubitPlaceholder(), noise=noise)
    ro = pq.declare('ro', 'BIT', len(code_register))
    pq += [MEASURE(qq, rr) for qq, rr in zip(code_register, ro)]

    return qvm.run(address_qubits(pq), trials=trials)


def simulate_code(kraus_operators, trials, error_code) -> int:
    """
    :param kraus_operators: The set of Kraus operators to apply as the noise model on the identity gate
    :param trials: The number of times to simulate the program
    :param error_code: The error code {bit_code, phase_code or shor} to use
    :return: The number of times the code did not correct back to the logical zero state for "trials" attempts
    """
    # Apply the error_code to some qubits and return back a Program pq
    qubit = QubitPlaceholder()
    pq, code_register = error_code(qubit, noise=None)
    ro = pq.declare("ro", "BIT", len(code_register))
    pq += [MEASURE(qq, rr) for qq, rr in zip(code_register, ro)]

    ################################################################
    # THIS CODE APPLIES THE NOISE FOR YOU
    kraus_ops = kraus_operators
    noise_data = Program()
    for qq in range(3):
        noise_data.define_noisy_gate("I", [qq], kraus_ops)
    pq = noise_data + pq
    ################################################################

    # Run the simulation trials times using the QVM and check how many times it did not work
    # return that as the score. E.g. if it always corrected back to the 0 state then it should return 0.    

    # Number of failures
    results = qvm.run(address_qubits(pq), trials=trials)

    score = sum([1 for row in results if row[0] > 0])  # Hacky way to count the number of trials that didn't correct back to 0
    return score
