#!/usr/bin/env python3
from net import Layer
from net import NeuralNet
import numpy as np

l = Layer(1, 1)
l2 = Layer(1, 1)
l3 = Layer(1, 1)
l4 = Layer(1, 1)

x_input = np.array([2])

r = 0.01
target = 10
x_input = 2

net = NeuralNet(layers = [l, l2, l3, l4], r = r, x_input = x_input, target = target)
predicted = net.forward()

print(f"{predicted}")

net.backpropagate()
print("=============")

net.backpropagate()
net.backpropagate()
net.backpropagate()
net.backpropagate()

predicted = net.forward()

print(f"{predicted}")


