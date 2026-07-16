#!/usr/bin/env python3
from engine import Layer
from engine import Activation_ReLU
from engine import NeuralNet 
import numpy as np


net = NeuralNet()
X = np.array([[2]])

prediction = net.forward(X)
print(prediction)


