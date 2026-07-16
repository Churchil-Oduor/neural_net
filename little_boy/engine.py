import numpy as np

class Layer_Dense:

    # initiating a layer
    def __init__(self, n_inputs, n_neurons):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.biases = np.random.rand(n_neurons)
        self.weights = 0.1 * np.random.rand(n_inputs, n_neurons)

    # forward
    def forward(self, inputs):
        q = Activation_ReLU()
        self.output = q.forward(np.dot(inputs, self.weights) + self.biases)
        

class Activation_ReLU:

    # forward pass
    def forward(self, inputs):
        return np.maximum(0, inputs)
        
        
class BackPropagation :

    def __init__(self, l_rate, y_true, weights):
        self.l_rate = l_rate
        self.y_true = y_true
        self.weights = weights


    # computes the new weight using gradient decent
    def new_weight(self, w, x):
        self.weight = w
        self.input = x
        loss = self.weight * self.input - self.y_true
        new_weight = self.weight - self.l_rate * 2 * loss * self.input
        return new_weight


    # handles back propagation
    def propagate(self, inputs):
       pass


class Softmax:
    pass

















