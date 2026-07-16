#!/usr/bin/env python3
from engine import Layer_Dense
from engine import Activation_ReLU
from engine import BackPropagation
import numpy as np

x = np.array([2])
layer = Layer_Dense(1, 4)
activation1 = Activation_ReLU()
layer.forward(x)
print(layer.output)
rate = 0.1
y_true = 10
#b = BackPropagation(rate, y_true)

#print(b.new_weight(4.4, 2))

