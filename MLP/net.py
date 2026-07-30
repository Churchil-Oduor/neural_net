import numpy as np



class MLP:

    """
    Multilayer Perceptron Class
    """

    def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2):
        """
        Constructor for the MLP. Takes the number of inputs,
        a variable number of hidden layers, and a number of outputs

        Args:
            num_inputs (int): Number of inputs
            hidden_layers (list): a list of inputs for the hidded layers
            num_outputs (int): Number of outputs
        """

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # generaric represetation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]

        weights = []

        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            weights.append(w)
        self.weights = weights



        activations = []
        for i in range (len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

    def forward_propagate(self, inputs):
        """Computes forward propagation of the network based on input signals

        Args:
            inputs (ndarray): Input signals
        Returns:
            activations (ndarray): Output values
        """


        # the input layer activatin is just the input itself
        activations = inputs
        self.activations[0] = inputs


        #iterating throught the network layers
        for i, w in enumerate(self.weights):
            # calculating matrix multiplication between previous activation 
            # and weight matrix

            net_inputs =np.dot(activations, w)
            activations = self._sigmoid(net_inputs)
            self.activations[i + 1] = activations

        return activations

    def back_propagate(self, error):
        for i in reversed(range(len(self.derivatives))):
            pass 



    def _sigmoid(self, x):
        """Sigmoid Activation function

        Args:
            x (float): Value to be processed
        Returns:
            y (float): Output
        """
        y = 1/(1 + np.exp(-x))
        return y

