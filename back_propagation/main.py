#!/usr/bin/env python3
from net2 import Layer
from net2 import NeuralNet
import numpy as np

np.random.seed(0)


layers = [
    Layer(1, 1),
    Layer(1, 1),
    Layer(1, 1),
    Layer(1, 1)
]

X = np.array([[3]])
y = np.array([[100]])

net = NeuralNet(layers, r=0.01)

for epoch in range(1000):
    prediction = net.forward(X)
    loss = net.compute_loss(y)

    net.backpropagate()

    if epoch % 100 == 0:
        print(epoch, loss)


