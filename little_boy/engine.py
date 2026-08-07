import numpy as np

class Layer:

    # initiating a layer
    def __init__(self, n_inputs, n_neurons):
        self.biases = np.zeros((1, n_neurons))
        self.weights = 0.1 * np.random.rand(n_inputs, n_neurons)

    # forward
    def forward(self, inputs):
        q = Activation_ReLU()
        self.inputs = inputs
        self.output = q.forward(np.dot(inputs, self.weights) + self.biases)

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims=True)

        #Passing gradient to previous layer
        self.dinputs = np.dot(dvalues, self.weights.T)


# ReLU activation function
class Activation_ReLU:

    # forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output
        


class NeuralNet:

    def __init__(self, layers = []):
        self.layers = layers
            

    def forward(self, input_x):
        nlayer = self.layers[0]
        nlayer.forward(input_x)

        for layer in self.layers[1:]:
            layer.forward(nlayer)
            nlayer = layer

        
        #self.layer1.forward(X)
        #self.layer2.forward(self.layer1.output)
        #self.layer3.forward(self.layer2.output)
        #self.layer4.forward(self.layer3.output)

        last_index = len(self.layers) - 1
        return self.layers[last_index].output


    #def backward(self, dloss):
     #   self.layer4.backward(dloss)
      #  self.layer3.backward(self.layer4.dinputs)
       # self.layer2.backward(self.layer3.dinputs)
       # self.layer1.backward(self.layer2.dinputs)
       # return self.layer1.dinputs
        
