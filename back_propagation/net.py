import numpy as np


class Layer:

    def __init__(self, n_inputs, n_neurons):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons

        self.old_weights = np.random.randn(n_inputs, n_neurons)
        self.new_weights = np.ones((n_inputs, n_neurons))
        
        self.f_weights = 1 # this will be update during backpropagation. represents the forward layer weights
                                
        self.biases = np.zeros((1, n_neurons))

    def forward(self, x_input):
        self.output = np.dot(x_input, self.old_weights) + self.biases
        return self.output


class Activation_ReLU:
    def forward(self, fward_values):
        self.output = np.maximum(0, fward_values)

class NeuralNet:

    def __init__(self, layers = [], r=0.1, x_input = 0, target = 0, y_pred = 0):
        self.layers = layers
        self.learning_rate = r
        self.x_input = x_input
        self.loss = y_pred - target

    def forward(self):

        f_layer = self.layers[0]
        f_layer.forward(self.x_input)

        for layer in self.layers[1:]:
            layer.forward(f_layer.output)
            f_layer = layer

        last_index = len(self.layers) - 1
        return self.layers[last_index].output


    def new_weight(self, old_w, grad):
        new_weight = old_w - self.learning_rate * grad
        return new_weight


    def back_prop_action(self, c_layer, p_layer):
        p_layer_output = p_layer.output
        dl_dw = 2 * self.loss * c_layer.f_weights
        p_layer.f_weights = dl_dw 
        full_grad_decent = dl_dw * p_layer_output
        c_layer.new_weights = self.new_weight(c_layer.old_weights, full_grad_decent)
        weight_tray = c_layer.old_weights
        c_layer.old_weights = c_layer.new_weights
        c_layer.new_weights = weight_tray

    def backpropagate(self):

        for i in range(len(self.layers) - 1, 0, -1):
            c_layer = self.layers[i] # current back layer
            p_layer = self.layers[i - 1] # previous back layer
            self.back_prop_action(c_layer, p_layer)


    def view_weights(self):
        for layer in self.layers:
            print(layer.output)

