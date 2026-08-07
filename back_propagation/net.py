import numpy as np

"""
class Layer defines a neural net layer
"""
class Layer:

    """Initializing a Layer

    Args:
        n_inputs (int): number of inputs
        n_neurons (int): number of neurons

    """
    def __init__(self, n_inputs=0, n_neurons=0):

        self.n_inputs = n_inputs
        self.n_neurons = n_neurons

        self.old_weights = np.random.randn(n_inputs, n_neurons)
        self.old_biases = np.random.randn(1, n_neurons)


    def forward(self, x_input):
        self.input = x_input
        self.z = np.dot(x_input, self.old_weights) + self.old_biases
        self.output = self.z
        return self.output



"""
class Activation_ReLU defines the ReLU activation.
"""

class Activation_ReLU:
    """Doing forward for the Activation ReLU.

    Args:
        x (int): output layer value to undergo activation.

    """
    def forward(self, x):
        self.input = x
        self.output = np.maximum(0, x)
        return self.output


    def backward(self, grad):
        grad = grad.copy()
        grad[self.input <= 0] = 0
        return grad



class NeuralNet:

    def __init__(self, layers =[], r=0.01):
        self.layers = layers
        self.learning_rate = r
        self.activations = [Activation_ReLU() for _ in range(len(layers) - 1)]


    def forward(self, x):
        current = x

        for i, layer in enumerate(self.layers):

            current = layer.forward(current)

            if i < len(self.layers) - 1:
                current = self.activations[i].forward(current)

        self.output = current
        return current


    def compute_loss(self, y_true):
        self.y_true = y_true
        self.loss = np.mean((self.output - y_true) ** 2)
        return self.loss


    def backpropagate(self):
        delta = 2 * (self.output - self.y_true)

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]

            dW = layer.input.T @ delta

            dB = np.sum(delta, axis=0, keepdims = True)

            old_weights = layer.old_weights.copy()
            old_biases = layer.old_biases.copy()

            layer.old_biases -= self.learning_rate * dB
            layer.old_weights -= self.learning_rate * dW

            if i == 0:
                break

            delta = delta @ old_weights.T
            delta = self.activations[i-1].backward(delta)



    def predict(self, x):
        return self.forward(x)
