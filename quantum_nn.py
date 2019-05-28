import pennylane as qml

from pennylane import numpy as np
from pennylane.optimize import GradientDescentOptimizer
from pennylane.ops import Hadamard, RX, CNOT
from shor_code_model import real_shor_code
from pyquil.api import WavefunctionSimulator


##############################################################


def add_noise(qubits: List[QubitPlaceholder]) -> List[QubitPlaceholder]:
    pass

def shor_encoding(qubit: QubitPlaceholder) -> List[QubitPlaceholder]:
    pass


def shor_decoding_model(qubits: List[QubitPlaceholder], weights) -> QubitPlaceholder:
    # Our neural network
    pass


def train(model_function, weights):
    # A training loop. Use GDO?
    # Construct our CNOt loss
    pass 

##############################################################


if __name__ == "__main__":
    weights = np.random.randn((7,7,7))  # Randomly initialize
    train(shor_decoding_model, weights)

    # Other things if necessary