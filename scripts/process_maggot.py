#%% [markdown]
# # Matching when including the contralateral connections
#%% [markdown]
# ## Preliminaries
#%%
import datetime
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.plot import adjplot, matched_stripplot, matrixplot
from numba import jit
from pkg.data import load_maggot_graph, load_matched
from pkg.io import OUT_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.match import BisectedGraphMatchSolver, GraphMatchSolver
from pkg.plot import method_palette, set_theme
from pkg.utils import get_paired_inds, get_paired_subgraphs, get_seeds
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon


FILENAME = "process_maggot"

DISPLAY_FIGS = True

OUT_PATH = OUT_PATH / FILENAME


def glue(name, var, **kwargs):
    default_glue(name, var, FILENAME, **kwargs)


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue(name, fig, figure=True)

    if not DISPLAY_FIGS:
        plt.close()


t0 = time.time()
set_theme()
rng = np.random.default_rng(8888)


#%% [markdown]
# ### Load the data


#%%
left_adj, left_nodes = load_matched("left")
right_adj, right_nodes = load_matched("right")
left_nodes["inds"] = range(len(left_nodes))
right_nodes["inds"] = range(len(right_nodes))
seeds = get_seeds(left_nodes, right_nodes)
all_nodes = pd.concat((left_nodes, right_nodes))
all_nodes["inds"] = range(len(all_nodes))

left_nodes.iloc[seeds[0]]["pair_id"]

assert len(left_nodes) == len(right_nodes)

#%%
mg = load_maggot_graph()
mg = mg.node_subgraph(all_nodes.index)
adj = mg.sum.adj

n = len(left_nodes)
left_inds = np.arange(n)
right_inds = np.arange(n) + n

#%%
