import random

import numpy as np

for i in range(3):
    random.seed(1)
    A = np.arange(10)
    random.shuffle(A)
    print(A)
