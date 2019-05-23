from typing import List
import numpy as np

from pyquil import Program
from pyquil.gates import MEASURE, I
from pyquil.gates import CNOT, H, X, Z, NOP
from pyquil.quil import address_qubits
from pyquil.quilatom import QubitPlaceholder
from pyquil.api import QVMConnection

##
############# YOU MUST COMMENT OUT THESE TWO LINES FOR IT TO WORK WITH THE AUTOGRADER
import subprocess
subprocess.Popen("/src/qvm/qvm -S > qvm.log 2>&1", shell=True)


# Do not change this SEED value you or your autograder score will be incorrect.
qvm = QVMConnection(random_seed=1337)


def bit_flip_channel(prob: float):
    noisy_I = np.sqrt(1-prob) * np.asarray([[1, 0], [0, 1]])
    noisy_X = np.sqrt(prob) * np.asarray([[0, 1], [1, 0]])
    return [noisy_I, noisy_X]


def phase_flip_channel(prob: float):
    noisy_I = np.sqrt(1-prob) * np.asarray([[1, 0], [0, 1]])
    noisy_Z = np.sqrt(prob) * np.asarray([[1, 0], [0, -1]])

    return [noisy_I, noisy_Z]


def depolarizing_channel(prob: float):
    noisy_I = np.sqrt(1-prob) * np.asarray([[1, 0], [0, 1]])
    noisy_X = np.sqrt(prob/3) * np.asarray([[0, 1], [1, 0]])
    noisy_Z = np.sqrt(prob/3) * np.asarray([[1, 0], [0, -1]])
    noisy_Y = np.sqrt(prob/3) * np.asarray([[0, -1j], [1j, 0]])

    # noisy_all = np.sqrt(prob/3) * (np.asarray([[0, 1], [1, 0]]) +  np.asarray([[1, 0], [0, -1]]) + np.asarray([[0, -1.j], [1.j, 0]]))
    # return [noisy_I, noisy_all]

    return [noisy_I, noisy_X, noisy_Y, noisy_Z]


def bit_code(qubit: QubitPlaceholder, noise=None, name=None) -> (Program, List[QubitPlaceholder]):

    ### Do your encoding step here

    # TODO: double check what they mean by how this list should get initialized
    code_register = [qubit, QubitPlaceholder(), QubitPlaceholder()]  # the List[QubitPlaceholder] of the qubits you have encoded into
    q0, q1, q2 = code_register

    # Apply the encoding step on the auixilary bits
    pq = Program(CNOT(q0, q1))  # the Program that does the encoding
    pq += CNOT(q0, q2)


    # DON'T CHANGE THIS CODE BLOCK. It applies the errors for simulations
    if noise is None:
        pq += [I(qq) for qq in code_register]
    else:
        pq += noise(code_register)


    ### Do your decoding and correction steps here

    #TODO: decoding step where you have to measure and go case by case on what was positive and what was negative

    # This will hold our measured ancilla bits
    if name:
        ro = pq.declare(name, memory_size=2)
    else:
        ro = pq.declare('antissa', memory_size=2)

    # encode the ancilla bits that we'll use to check the parity that we'll later use to make our correction
    # decision
    a1 = QubitPlaceholder()
    a2 = QubitPlaceholder()

    pq += CNOT(q0, a1)
    pq += CNOT(q1, a1)

    pq += CNOT(q1, a2)
    pq += CNOT(q2, a2)


    # A draft of somethign that might measure everything together this measures everything together
    # qubit_to_bit_tuples = [(code_register[0], ro[0]),  (code_register[1], ro[1]), (code_register[2], ro[2]) ]
    # pq.measure_all(qubit_to_bit_tuples)

    # Attempt at measureing things conveninetly
    # qubit_to_bit_tuples = [(a1, ro[0]),  (a2, ro[1]) ]
    # pq.measure_all(qubit_to_bit_tuples)

    # Attempt doing it manually
    pq += MEASURE(a1, ro[0])
    pq += MEASURE(a2, ro[1])


    # Now go through the different branches that we'll use
    # Make the program structure no bit change
    first_one_branch = Program()
    first_one_branch.if_then(ro[1], X(q1), X(q0) )


    first_zero_branch = Program()
    first_zero_branch.if_then(ro[1], X(q2), NOP)
    # End making branches


    # Now launch the driver program into going through the cases
    pq.if_then(ro[0], first_one_branch, first_zero_branch)


    return pq, code_register


def phase_code(qubit: QubitPlaceholder, noise=None) -> (Program, List[QubitPlaceholder]):
    ### Do your encoding step here
    code_register = [qubit, QubitPlaceholder(),
                     QubitPlaceholder()]  # the List[QubitPlaceholder] of the qubits you have encoded into
    q0, q1, q2 = code_register

    # Apply the encoding step on the redundent bits
    pq = Program(CNOT(q0, q1))  # the Program that does the encoding
    pq += CNOT(q0, q2)

    # Apply Haddamard gates to convert phase-flips to look like bit-flips
    pq += H(q0)
    pq += H(q1)
    pq += H(q2)

    # DON'T CHANGE THIS CODE BLOCK. It applies the errors for simulations
    if noise is None:
        pq += [I(qq) for qq in code_register]
    else:
        pq += noise(code_register)

    ### Do your decoding and correction steps here


    # Now the noise has been applied so let's convert back to the original basis
    pq += H(q0)
    pq += H(q1)
    pq += H(q2)


    # This will hold our measured ancilla bits
    ro = pq.declare('antissa', memory_size=2)

    # encode the ancilla bits that we'll use to check the parity that we'll later use to make our correction
    # decision
    a1 = QubitPlaceholder()
    a2 = QubitPlaceholder()

    pq += CNOT(q0, a1)
    pq += CNOT(q1, a1)

    pq += CNOT(q1, a2)
    pq += CNOT(q2, a2)


    # Read the ancilla bits
    pq += MEASURE(a1, ro[0])
    pq += MEASURE(a2, ro[1])

    # Now go through the different branches that we'll use
    # Make the program structure no bit change
    first_one_branch = Program()
    first_one_branch.if_then(ro[1], Program(H(q1)) + Program(Z(q1)) + Program(H(q1)), Program(H(q0)) + Program(Z(q0)) + Program(H(q0)))

    first_zero_branch = Program()
    first_zero_branch.if_then(ro[1], Program(H(q2)) + Program(Z(q2)) + Program(H(q2)), NOP)
    # End making branches


    # Now launch the driver program into going through the cases
    pq.if_then(ro[0], first_one_branch, first_zero_branch)

    return pq, code_register



# adds an encoding step to the given program
def encoding_helper(q0, q1, q2, program):
    program += CNOT(q0, q1)  # the Program that does the encoding
    program += CNOT(q0, q2)

    return program

def apply_H_to_list(qbit_list, program):
    for curr_qbit in qbit_list:
        program += H(curr_qbit)

    return program

def find_bit_parity(q0, q1, q2, classical_mem_name, pq):
    ro = pq.declare(classical_mem_name, memory_size=2)

    a1 = QubitPlaceholder()
    a2 = QubitPlaceholder()

    pq += CNOT(q0, a1)
    pq += CNOT(q1, a1)

    pq += CNOT(q1, a2)
    pq += CNOT(q2, a2)

    # apply H to ancilla bit
    # pq += H(a1)
    # pq += H(a2)


    # Attempt doing it manually
    pq += MEASURE(a1, ro[0])
    pq += MEASURE(a2, ro[1])


    # Now go through the different branches that we'll use
    # Make the program structure no bit change
    first_one_branch = Program()
    first_one_branch.if_then(ro[1], X(q1), X(q0) )


    first_zero_branch = Program()
    first_zero_branch.if_then(ro[1], X(q2), NOP)
    # End making branches


    # Now launch the driver program into going through the cases
    pq.if_then(ro[0], first_one_branch, first_zero_branch)

    return pq


def apply_phase_correction(qbit_list):
    # apply the first H
    pq = Program()
    for qbit in qbit_list:
        pq += H(qbit)
        pq += Z(qbit)
        pq += H(qbit)


    return pq

def apply_phase_bit_parity(first_block, second_block, third_block, classical_mem_name, pq):

    ro = pq.declare(classical_mem_name, memory_size=2)

    # encode the ancilla bits that we'll use to check the parity that we'll later use to make our correction
    # decision
    a1 = QubitPlaceholder()
    a2 = QubitPlaceholder()


    first_list = first_block + second_block
    second_list = second_block + third_block
    for qbit in first_list:
        pq += CNOT(qbit, a1)

    for qbit in second_list:
        pq += CNOT(qbit, a2)

    # q_list parameter just holds the leading qbits of each block

    # Read the ancilla bits
    pq += MEASURE(a1, ro[0])
    pq += MEASURE(a2, ro[1])

    # Now go through the different branches that we'll use
    # Make the program structure no bit change
    first_one_branch = Program()
    first_one_branch.if_then(ro[1], apply_phase_correction(second_block), apply_phase_correction(first_block))

    first_zero_branch = Program()
    first_zero_branch.if_then(ro[1], apply_phase_correction(third_block), NOP)
    # End making branches


    # Now launch the driver program into going through the cases
    pq.if_then(ro[0], first_one_branch, first_zero_branch)

    return pq


def shor(qubit: QubitPlaceholder, noise=None) -> (Program, List[QubitPlaceholder]):
    # leading_block_bits = [qubit, QubitPlaceholder(), QubitPlaceholder()]
    # second_block = [QubitPlaceholder(), QubitPlaceholder(), QubitPlaceholder()]
    # third_block = [QubitPlaceholder(), QubitPlaceholder(), QubitPlaceholder()]

    # Apply the first encoding that link up the blocks together
    # pq = encoding_helper(leading_block_bits[0], leading_block_bits[1], leading_block_bits[2], pq)

    # Now apply the H-gate to the first qbit of each block so that now they're in the phase-basis
    # pq = apply_H_to_list([leading_block_bits[0], leading_block_bits[1], leading_block_bits[2]], pq)


    pq, leading_block_bits = phase_code(qubit,noise)


    flip_pq_1, first_block = bit_code(leading_block_bits[0], noise, name="first")
    flip_pq_2, second_block = bit_code(leading_block_bits[1], noise, name="second")
    flip_pq_3, third_block = bit_code(leading_block_bits[2], noise, name="third")

    pq = pq + flip_pq_1 + flip_pq_2 + flip_pq_3

    return pq, first_block + second_block + third_block


def shor_alt(qubit: QubitPlaceholder, noise=None) -> (Program, List[QubitPlaceholder]):
    # Note that in order for this code to work properly, you must build your Shor code using the phase code and
    # bit code methods above


    first_block = [qubit, QubitPlaceholder(), QubitPlaceholder()]
    second_block = [QubitPlaceholder(), QubitPlaceholder(), QubitPlaceholder()]
    third_block = [QubitPlaceholder(), QubitPlaceholder(), QubitPlaceholder()]


    # Apply the first encoding that link up the blocks together
    pq = Program()
    pq = encoding_helper(first_block[0], second_block[0], third_block[0], pq)

    # Now apply the H-gate to the first qbit of each block so that now they're in the phase-basis
    pq = apply_H_to_list([first_block[0], second_block[0], third_block[0]], pq)

    # Now encode together bits of each block respectivly
    pq = encoding_helper(first_block[0], first_block[1], first_block[2], pq)
    pq = encoding_helper(second_block[0], second_block[1], second_block[1], pq)
    pq = encoding_helper(third_block[0], third_block[1], third_block[2], pq)

    #END encoding; apply noise now
    # DON'T CHANGE THIS CODE BLOCK. It applies the errors for simulations
    if noise is None:
        pq += [I(qq) for qq in first_block + second_block + third_block]
    else:
        pq += noise(first_block + second_block + third_block)

    # Now apply the parity operations to check for bit-flips
    pq = find_bit_parity(first_block[0], first_block[1], first_block[2], "first_block_bit_parity", pq)
    pq = find_bit_parity(second_block[0], second_block[1], second_block[2], "second_block_bit_parity", pq)
    pq = find_bit_parity(third_block[0], third_block[1], third_block[2], "third_block_bit_parity", pq)

    # now apply H-gate again to all qbits
    pq = apply_H_to_list(first_block + second_block + third_block, pq)

    # now apply phase shift correction
    pq = apply_phase_bit_parity(first_block, second_block, third_block, "phase_parity", pq)

    return pq, first_block + second_block + third_block


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
    pq, code_register = error_code(QubitPlaceholder(), noise=None)

    # THIS CODE APPLIES THE NOISE FOR YOU
    kraus_ops = kraus_operators
    noise_data = Program()
    for qq in range(3):
        noise_data.define_noisy_gate("I", [qq], kraus_ops)
    pq = noise_data + pq

    # Run the simulation trials times using the QVM and check how many times it did not work
    # return that as the score. E.g. if it always corrected back to the 0 state then it should return 0.

    recovered_bits = pq.declare('ro', 'BIT', len(code_register))
    pq += [MEASURE(qq, rr) for qq, rr in zip(code_register, recovered_bits)]
    trial_results = qvm.run(address_qubits(pq), trials=trials)

    print("trial_results print")
    print(trial_results)

    # Num of times we didn't recovered a full (0, 0, 0) result
    score = 0

    for output_set in trial_results:
        if np.sum(output_set) != 0:
            score += 1

    print("score")
    print(score)

    return score

# Use this as the noise() function parameter to manually just apply a phase shift to the first qbit only
def apply_z_to_first(code_register):
    return Program(Z(code_register[0]))


# BEGIN MAIN PROGRAM
# running the simulation for the bit_channel
# num_trials = 10
#
# # Run a bit_code
# simulate_code(bit_flip_channel(0.0), num_trials, bit_code)

# # Run phase_code
# print("Phase correction")
# simulate_code(phase_flip_channel(0.0), num_trials, phase_code)
#
# Run Schor code
# print("Schor code correction")
# simulate_code(depolarizing_channel(0), num_trials, shor_alt)

