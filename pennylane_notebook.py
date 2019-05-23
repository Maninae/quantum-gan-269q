import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import GradientDescentOptimizer
from pennylane.ops import Hadamard, RX, CNOT
####################################################################################


dev = qml.device('default.qubit', wires=2)

def quantum_gradient_descent_demo():

    # Declare quantum circuit
    @qml.qnode(dev)
    def circuit(theta1, theta2):
        Hadamard(wires=1)
        RX(theta1, wires=0)
        RX(theta2, wires=1)
        CNOT(wires=[0,1])
        return qml.expval.PauliZ(wires=0)
    """    To design other circuits, we have
            Pennylane other gates:      qml.ops.qubit.all_ops
            Pennylane measurement ops:  qml.expval.qubit.all_ops
    """

    # We can evaluate the circuit at any value of the parameters
    print(circuit(0, 0))
    for theta1, theta2 in np.random.random((5,2)):
        print(circuit(theta1, theta2))
    """ 0.9999999999999998
        0.5955486271678074
        0.9641638120309893... [5 total]
    """

    grad_circuit = qml.grad(circuit, argnum=[0,1])
    # We can evaluate the circuit gradient at any value of the parameters
    for theta1, theta2 in np.random.random((5,2)):
        print("Value: ({:3f},{:3f}); Gradient={}".format(theta1, theta2, 
                                                 np.stack(grad_circuit(theta1, theta2))))
    """ Value: (0.194877,0.991503); Gradient=[-1.93645932e-01 -5.55111512e-17]
        Value: (0.642381,0.922974); Gradient=[-0.5991039  0.       ]
        Value: (0.102674,0.684560); Gradient=[-1.02494172e-01 -1.66533454e-16]
        Value: (0.041783,0.109417); Gradient=[-4.17708740e-02  1.11022302e-16]
        Value: (0.039356,0.880646); Gradient=[-3.93455645e-02 -1.66533454e-16]
    """

    eta = 0.1
    opt = GradientDescentOptimizer(eta)
    # Other pennylane optimizers:   qml.optimize.__all__

    init_val = np.random.random(2)
    new_val = opt.step(circuit, init_val)
    print("Initial value:", init_val)
    print("Value after one step:", new_val)


    # Define a classical cost function which post-processes the circuit's output
    target = 0.33
    def cost(weights):
        expval = circuit(weights[0], weights[1])
        error = np.abs(expval - target) ** 2 + weights[2] ** 2
        return error

    # Evaluate cost at a random starting point
    weights = np.random.random(3)
    print(cost(weights))

    # Implement gradient descent over 100 steps
    for step in range(100):
        weights = opt.step(cost, weights)
        if step % 10 == 0:
            print("Step {}: Cost={}".format(step, cost(weights)))

####################################################################################

gan_dev = qml.device('default.qubit', wires=3)

def quantum_gan_demo():

    # Generate "real" (i.e. fixed) data
    def real(phi, theta, omega):
        qml.Rot(phi, theta, omega, wires=0)

    # Note: the structure of these circuits is more-or-less chosen arbitrarily. There is 
# nothing particular about these choices of generator or discriminator

    def generator(w):
        qml.RX(w[0], wires=0)
        qml.RX(w[1], wires=1)
        qml.RY(w[2], wires=0)
        qml.RY(w[3], wires=1)
        qml.RZ(w[4], wires=0)
        qml.RZ(w[5], wires=1)
        qml.CNOT(wires=[0,1])
        qml.RX(w[6], wires=0)
        qml.RY(w[7], wires=0)
        qml.RZ(w[8], wires=0)
        
    def discriminator(w):
        qml.RX(w[0], wires=0)
        qml.RX(w[1], wires=2)
        qml.RY(w[2], wires=0)
        qml.RY(w[3], wires=2)
        qml.RZ(w[4], wires=0)
        qml.RZ(w[5], wires=2)
        qml.CNOT(wires=[1,2])
        qml.RX(w[6], wires=2)
        qml.RY(w[7], wires=2)
        qml.RZ(w[8], wires=2)

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
    quantum_gan_demo()


