import numpy as np


class Layer:

    def __init__(self, n_inputs, n_neurons):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, x_input):
        self.output = np.dot(x_input, self.weights) + self.biases
        return self.output

class Activation_ReLU:
    def forward(self, fward_values):
        self.output = np.maximum(0, fward_values)

class NeuralNet:

    def __init__(self, layers = [], r=0.1, x_input = 0, target = 0):
        self.layers = layers
        self.learning_rate = r
        self.target = target
        self.x_input = x_input

    def forward(self):

        f_layer = self.layers[0]
        f_layer.forward(self.x_input)

        for layer in self.layers[1:]:
            layer.forward(f_layer.output)
            f_layer = layer

        last_index = len(self.layers) - 1
        return self.layers[last_index].output


    def new_weight(self, w, y_pred):
        loss = y_pred - self.target
        gradient = 2 * loss * self.x_input
        n_weight = w - self.learning_rate * gradient
        return n_weight

    def backpropagate(self):

        for i in range(len(self.layers) - 1, 0, -1):
            current = self.layers[i]
            previous = self.layers[i - 1]

            old_weight = previous.weights
            y_pred = current.output

            n_w = self.new_weight(old_weight, y_pred)
            previous.weights = n_w
