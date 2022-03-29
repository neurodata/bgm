import numpy as np

from .bisected_gm import BisectedGraphMatchSolver


class GraphMatchSolver(BisectedGraphMatchSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.AB = np.zeros_like(self.AB)
        self.BA = np.zeros_like(self.BA)
