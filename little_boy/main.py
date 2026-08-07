#!/usr/bin/env python3
from engine import Layer
from engine import Activation_ReLU
from engine import NeuralNet 
import numpy as np


X = np.array([[2]])
layer = Layer(1, 1)

net = NeuralNet([Layer(1, 1), Layer(1, 1)])
#net.forward(X)
print(net.forward(X))



