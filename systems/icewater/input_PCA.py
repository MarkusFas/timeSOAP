from random import shuffle

from src.setup.launcher import run_simulation
from src.setup.input_check import setup_simulation



shuffle(selected_atoms)
train_atoms = selected_atoms[:20] # take only 20 random atoms
test_atoms = selected_atoms[30:50] # take only 20 random atoms