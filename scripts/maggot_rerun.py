#%%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pkg.data import load_split_connectome
from pkg.io import OUT_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.utils import get_hemisphere_indices

from graspologic.match import graph_match

FILENAME = "maggot_rerun"

DISPLAY_FIGS = True

OUT_PATH = OUT_PATH / FILENAME


def glue(name, var, **kwargs):
    default_glue(name, var, FILENAME, **kwargs)


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue(name, fig, figure=True)

    if not DISPLAY_FIGS:
        plt.close()


rng = np.random.default_rng(8888)

#%% [markdown]
# ## Load processed data, run matching experiment
#%%


dataset = "maggot_subset"

adj, nodes = load_split_connectome(dataset)

left_inds, right_inds = get_hemisphere_indices(nodes)
A = adj[left_inds][:, left_inds]
B = adj[right_inds][:, right_inds]
AB = adj[left_inds][:, right_inds]
BA = adj[right_inds][:, left_inds]

indices_A, indices_B, score, misc = graph_match(A, B, AB=AB, BA=BA, rng=rng)


#%%
(indices_A == np.arange(len(left_inds))).all()
acc = (indices_B == np.arange(len(right_inds))).mean()
acc

#%%
node_ids_left = nodes.iloc[left_inds].index
node_ids_right = nodes.iloc[right_inds].index
matched_node_ids_left = pd.Series(node_ids_left[indices_A], name="left_id")
matched_node_ids_right = pd.Series(node_ids_right[indices_B], name="right_id")
is_correct = pd.Series(indices_B == np.arange(len(right_inds)), name="is_correct")

match_df = pd.concat(
    (matched_node_ids_left, matched_node_ids_right, is_correct), axis=1
)

match_df.to_csv(OUT_PATH / "match_df.csv", index=False)
