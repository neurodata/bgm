#%%
import datetime
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pkg.data import DATA_PATH, load_maggot_graph, load_matched
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.plot import set_theme
from pkg.utils import get_seeds
from pkg.utils import select_lateral_nodes, ensure_connected
from pkg.pymaid import start_instance
import pymaid

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
all_left = mg.nodes[mg.nodes["left"]].index
all_right = mg.nodes[mg.nodes["right"]].index

m_ll = mg.node_subgraph(all_left, all_left).summary_statistics.loc["sum", "n_edges"]
m_rr = mg.node_subgraph(all_right, all_right).summary_statistics.loc["sum", "n_edges"]
m_lr = mg.node_subgraph(all_left, all_right).summary_statistics.loc["sum", "n_edges"]
m_rl = mg.node_subgraph(all_right, all_left).summary_statistics.loc["sum", "n_edges"]

m_contra = m_lr + m_rl
m_ipsi = m_ll + m_rr
p_contra = m_contra / (m_ipsi + m_contra)
glue("p_contra", p_contra, form="2.0f%")

#%%
mg = mg.node_subgraph(all_nodes.index)
adj = mg.sum.adj
nodes = all_nodes
adj_df = pd.DataFrame(adj.astype(int), index=all_nodes.index, columns=all_nodes.index)

#%%
start_instance()

#%%


def get_indicator_from_annotation(annot_name, filt=None):
    ids = pymaid.get_skids_by_annotation(annot_name.replace("*", "\*"))
    if filt is not None:
        name = filt(annot_name)
    else:
        name = annot_name
    indicator = pd.Series(
        index=ids, data=np.ones(len(ids), dtype=bool), name=name, dtype=bool
    )
    return indicator


annot_df = pymaid.get_annotated("papers")
series_ids = []

for annot_name in annot_df["name"]:
    print(annot_name)
    indicator = get_indicator_from_annotation(annot_name)
    series_ids.append(indicator)
annotations = pd.concat(series_ids, axis=1, ignore_index=False).fillna(False)


#%%
# published_types = [
#     "LON",
#     "mPN",
#     "RGN",
#     "uPN",
#     "MBIN",
#     "FFN",
#     "FB2N",
#     "tPN",
#     "KC",
#     "dVNC;RGN",
#     "MBON",
#     "FAN",
#     "pLN",
#     "FBN",
#     "cLN",
#     "bLN",
#     "vPN",
#     "APL",
#     "motor",
#     "sens",
#     "keystone",
# ]  # A00c?

# nodes = nodes[nodes["class1"].isin(published_types)].copy()
nodes = nodes[nodes.index.isin(annotations.index)]
adj_df = adj_df.reindex(index=nodes.index, columns=nodes.index)

nodes["pair"] = nodes["pair_id"]


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
