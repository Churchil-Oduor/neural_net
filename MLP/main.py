#!/usr/bin/env python3
from net import MLP
import numpy as np
from random import random
import math as mth



f = mth.pi / 180

if __name__ == "__main__":

    mlp = MLP(2, [5], 1)
    l_rate = 0.1

    # create dummy data
    inputs = np.array([[random()/2 for _ in range(2)] for _ in range(100)])
    targets = np.array([(i[0] + i[1]) for i in inputs])

    mlp.train(inputs, targets, 100, l_rate)
   # testing
    input = np.array([0.2, 0.2])
    output = mlp.forward_propagate(input)
    print()
    print()

    print("Our network believes that sum of {} + {} = {}".format(input[0], input[1], output))


