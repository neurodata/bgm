#%%
import datetime
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.plot import adjplot, matched_stripplot, matrixplot
from pkg.data import DATA_PATH, load_maggot_graph, load_matched
from pkg.io import OUT_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.plot import set_theme
from pkg.utils import get_seeds

FILENAME = "process_maggot"

DISPLAY_FIGS = True

OUT_PATH = DATA_PATH / "processed_split"


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

nodes = all_nodes
adj_df = pd.DataFrame(adj.astype(int), index=all_nodes.index, columns=all_nodes.index)


#%%
published_types = [
    "LON",
    "mPN",
    "RGN",
    "uPN",
    "MBIN",
    "FFN",
    "FB2N",
    "tPN",
    "KC",
    "dVNC;RGN",
    "MBON",
    "FAN",
    "pLN",
    "FBN",
    "cLN",
    "bLN",
    "vPN",
    "APL",
    "motor",
    "sens",
    "keystone",
]  # A00c?

nodes = nodes[nodes["class1"].isin(published_types)].copy()
adj_df = adj_df.reindex(index=nodes.index, columns=nodes.index)

nodes["pair"] = nodes["pair_id"]

from pkg.utils import select_lateral_nodes, ensure_connected

adj_df, nodes, removed_nonlateral = select_lateral_nodes(adj_df, nodes)
# then ensure the network is fully connected
adj_df, nodes, removed_lcc = ensure_connected(adj_df, nodes)
# then remove any nodes whose partner got removed by that process
adj_df, nodes, removed_partner_lcc = select_lateral_nodes(adj_df, nodes)
# REPEAT in case this removal of partners causes disconnection
adj_df, nodes, removed_lcc2 = ensure_connected(adj_df, nodes)
adj_df, nodes, removed_partner_lcc2 = select_lateral_nodes(adj_df, nodes)

#%%
g = nx.from_pandas_adjacency(adj_df, create_using=nx.DiGraph)

nx.write_edgelist(
    g, OUT_PATH / "maggot_subset_edgelist.csv", delimiter=",", data=["weight"]
)

nodes.to_csv(OUT_PATH / "maggot_subset_nodes.csv")

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
