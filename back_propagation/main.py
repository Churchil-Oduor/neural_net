#!/usr/bin/env python3
from net import Layer
from net import NeuralNet
import numpy as np

np.random.seed(0)


layers = [
    Layer(1, 10),
    Layer(10, 20),
    Layer(20, 10),
    Layer(10, 1)
]

X = np.array([[45]])
y = np.array([[0.70710678118]])

net = NeuralNet(layers, r=0.1)

for i in range(200):
    net.forward(X)
    net.compute_loss(y)
    net.backpropagate()
    print(net.predict(X), net.compute_loss(y))

print("=================")

X = np.array([[0]])
y = np.array([[1]])

for i in range(200):
    net.forward(X)
    net.compute_loss(y)
    net.backpropagate()
    print(net.predict(X), net.compute_loss(y))


