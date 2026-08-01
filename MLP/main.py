#!/usr/bin/env python3
from net import MLP
import numpy as np
from random import random
import math as mth



f = mth.pi / 180

if __name__ == "__main__":

    mlp = MLP(2, [5, 10, 10, 10, 5], 1)
    l_rate = 0.1

    # create dummy data
    inputs = np.array([[random()*100 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[np.cos((i[0] + i[1]) * f) ]for i in inputs])

    mlp.train(inputs, targets, 50, l_rate)

    #testing
    input = np.array([20, 25])
    output = mlp.forward_propagate(input)
    print()
    print()

    print("Our network believes that {} + {} = {}".format(input[0], input[1], output))


