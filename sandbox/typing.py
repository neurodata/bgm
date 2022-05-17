#%%
from giskard.match import GraphMatchSolver
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.zeros((3, 3))
solver = GraphMatchSolver([A, B], A)
