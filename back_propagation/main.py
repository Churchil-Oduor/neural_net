#!/usr/bin/env python3
from net import Layer
from net import NeuralNet
import numpy as np

np.random.seed(0)


layers = [
    Layer(1, 1),
    Layer(1, 1),
    Layer(1, 1),
    Layer(1, 1)
]

X = np.array([[2]])
y = np.array([[12]])

net = NeuralNet(layers, r=0.001)

for i in range(100):
    print(net.forward(X))
    net.compute_loss(y)
    net.backpropagate()
    print(net.predict(X))

