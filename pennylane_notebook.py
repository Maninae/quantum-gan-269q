import pennylane as qml

from pennylane import numpy as np
from pennylane.optimize import GradientDescentOptimizer
from pennylane.ops import Hadamard, RX, CNOT
from shor_code_model import real_shor_code
from pyquil.api import WavefunctionSimulator

####################################################################################


# dev = qml.device('default.qubit', wires=2)

gan_dev = qml.device('default.qubit', wires=10)

    # qml.device('12q-pyqvm', wires=9)

def quantum_gan_model():

    num_qbits = 9
    workspace_wire = 9


    # TODO: Maybe make a "ShorCode" class for this part
    # Change this to give the data from the output of the Shore Code algorithm
    # Generate "real" (i.e. fixed) data
    def real(phi, theta, omega):

        pyquil_qbits = real_shor_code()


        # qml.Rot(phi, theta, omega, wires=0)

    # Note: the structure of these circuits is more-or-less chosen arbitrarily. There is 
# nothing particular about these choices of generator or discriminator

    def generator(w):

        # Wires 0 to 8 to hold generator output

        # entagle qbits 0 to 8
        for i in range(num_qbits):
            first = i
            second = workspace_wire
            qml.RX(w[0], wires=first)
            qml.RX(w[1], wires=second)
            qml.RY(w[2], wires=first)
            qml.RY(w[3], wires=second)
            qml.RZ(w[4], wires=first)
            qml.RZ(w[5], wires=second)
            qml.CNOT(wires=[first,second])
            qml.RX(w[6], wires=first)
            qml.RY(w[7], wires=first)
            qml.RZ(w[8], wires=first)
        
    def discriminator(w):
        # entagle qbits 0 to 8

        for i in range(num_qbits):
            curr_input_wire = i
            qml.RX(w[0], wires=curr_input_wire)
            qml.RX(w[1], wires=workspace_wire)
            qml.RY(w[2], wires=curr_input_wire)
            qml.RY(w[3], wires=workspace_wire)
            qml.RZ(w[4], wires=curr_input_wire)
            qml.RZ(w[5], wires=workspace_wire)
            qml.CNOT(wires=[1,workspace_wire])
            qml.RX(w[6], wires=workspace_wire)
            qml.RY(w[7], wires=workspace_wire)
            qml.RZ(w[8], wires=workspace_wire)

    # Create 2 QNodes (i.e. circuits) for the Gen and Disc.
    @qml.qnode(gan_dev)
    def real_disc_circuit(phi, theta, omega, disc_weights):  # Run Discriminator on real data
        real(phi, theta, omega)
        discriminator(disc_weights)
        return qml.expval.PauliZ(2)

    @qml.qnode(gan_dev)
    def gen_disc_circuit(gen_weights, disc_weights):         # Run Discriminator on FAKE data
        generator(gen_weights)
        discriminator(disc_weights)
        return qml.expval.PauliZ(2)

    """ All of the losses! 4 components: 
          (1) Probability discriminator says real data is real
          (2) Probability discriminator says FAKE data is real
        Disc loss: (2) - (1)
            Tries to maximize prob of telling real=real, FAKE=FAKE
        Gen loss:  -(2)
            Tries to maximize prob that Disc says FAKE data is real
    """
    def prob_real_true(disc_weights):
        true_disc_output = real_disc_circuit(phi, theta, omega, disc_weights)
        # convert to probability
        prob_real_true = (true_disc_output + 1) / 2
        return prob_real_true

    def prob_fake_true(gen_weights, disc_weights):
        fake_disc_output = gen_disc_circuit(gen_weights, disc_weights)
        # convert to probability
        prob_fake_true = (fake_disc_output + 1) / 2
        return prob_fake_true # generator wants to minimize this prob

    def disc_cost(disc_weights):
        cost = prob_fake_true(gen_weights, disc_weights) - prob_real_true(disc_weights) 
        return cost

    def gen_cost(gen_weights):
        return -prob_fake_true(gen_weights, disc_weights)

    #.....................#

    # Real data params! FIXED at beginning here
    phi = np.pi / 6
    theta = np.pi / 2
    omega = np.pi / 7
    np.random.seed(0)
    eps = 1e-2

    # Generator G and Discriminator D weights

    #TODO: consider a different initialization for the generator weights here
    gen_weights = np.array([np.pi] + [0] * 8) + np.random.normal(scale=eps, size=[9])
    disc_weights = np.random.normal(size=[9])
    
    # Create an optimizer
    opt = GradientDescentOptimizer(0.1)

    # Optimize D, fix G
    for it in range(50):
        disc_weights = opt.step(disc_cost, disc_weights) 
        cost = disc_cost(disc_weights)
        if it % 5 == 0:
            print("Step {}: cost = {}".format(it+1, cost))

    # Discriminator good now, generator pretty poor
    # Close to 1, 0 respectively.
    print("After D trained, prob_real_true:", prob_real_true(disc_weights))
    print("After D trained, prob_fake_true:", prob_fake_true(gen_weights, disc_weights))

    # Optimize G, fix D
    for it in range(200):
        gen_weights = opt.step(gen_cost, gen_weights)
        cost = -gen_cost(gen_weights)
        if it % 5 == 0:
            print("Step {}: cost = {}".format(it, cost))

    # Should be close to 1! Generator has fooled discriminator
    print("After G trained, prob_real_true:", prob_real_true(disc_weights))
    print("After G trained, prob_fake_true:", prob_fake_true(gen_weights, disc_weights))


####################################################################################

if __name__ == "__main__":
    # quantum_gradient_descent_demo()
    quantum_gan_model()


