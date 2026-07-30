#!/usr/bin/env python3
from net import MLP
import numpy as np


mlp = MLP(1)
x = np.array([2])
print(len(mlp.weights))

