import torch
from metatensor.torch import Labels
import numpy as np


labels = Labels(
            names=["system", "atom", "type"],
            values=torch.tensor([(0, 1, 8), (0, 2, 1), (0, 5, 1)]),
            )

positions = np.random.random((3, 3))
print(labels)

selected_atoms = Labels(
    names=labels.names,
    values=labels.values[2:,:],   
)
print(selected_atoms)