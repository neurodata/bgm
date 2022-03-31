#%%
import datetime
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from graspologic.plot import adjplot
from graspologic.utils import largest_connected_component
from pkg.data import DATA_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.plot import set_theme


FILENAME = "process_c_elegans"

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


#%%


def load_adjacency(path):
    adj_df = pd.read_csv(path, index_col=0).fillna(0)
    node_ids = np.union1d(adj_df.index, adj_df.columns)
    adj_df = adj_df.reindex(index=node_ids, columns=node_ids).fillna(0)
    adj_df = pd.DataFrame(
        data=adj_df.values.astype(int), index=adj_df.index, columns=adj_df.columns
    )
    return adj_df


def create_node_data(node_ids):
    exceptions = ["vBWM", "dgl", "dBWM"]
    node_rows = []

    for node_id in node_ids:
        is_sided = True
        if not ((node_id[-1] == "L") or (node_id[-1] == "R")):
            is_exception = False
            for exception in exceptions:
                if exception in node_id:
                    is_exception = True
            if not is_exception:
                is_sided = False

        if is_sided:
            # node_id_no_side = node_id.strip("0123456789")
            left_pos = node_id.rfind("L")
            right_pos = node_id.rfind("R")
            is_right = bool(np.argmax((left_pos, right_pos)))
            side_indicator_loc = right_pos if is_right else left_pos
            node_pair = node_id[:side_indicator_loc] + node_id[side_indicator_loc + 1 :]
            hemisphere = "R" if is_right else "L"
            node_rows.append(
                {"node_id": node_id, "pair": node_pair, "hemisphere": hemisphere}
            )

    nodes = pd.DataFrame(node_rows).set_index("node_id")

    return nodes


def select_lateral_nodes(adj_df, nodes):

    counts = nodes.groupby("pair").size()
    singleton_classes = counts[counts != 2].index

    removed = nodes[nodes["pair"].isin(singleton_classes)]

    nodes = nodes[~nodes["pair"].isin(singleton_classes)]

    nodes = nodes.sort_values(["hemisphere", "pair"])
    left_nodes = nodes[nodes["hemisphere"] == "L"]
    right_nodes = nodes[nodes["hemisphere"] == "R"]
    assert (left_nodes["pair"].values == right_nodes["pair"].values).all()

    adj_df = adj_df.reindex(index=nodes.index, columns=nodes.index)

    return adj_df, nodes, removed


def ensure_connected(adj_df, nodes):
    adj_df = adj_df.reindex(index=nodes.index, columns=nodes.index)
    adj = adj_df.values

    adj_lcc, inds = largest_connected_component(adj, return_inds=True)

    removed = nodes[~nodes.index.isin(nodes.index[inds])]
    nodes = nodes.iloc[inds]

    adj_df = pd.DataFrame(data=adj_lcc, index=nodes.index, columns=nodes.index)

    return adj_df, nodes, removed


def split_nodes(nodes):
    nodes = nodes.sort_values(["hemisphere", "pair"])
    left_nodes = nodes[nodes["hemisphere"] == "L"]
    right_nodes = nodes[nodes["hemisphere"] == "R"]
    assert (left_nodes["pair"].values == right_nodes["pair"].values).all()
    return left_nodes, right_nodes


#%%

for sex in ["male", "herm"]:
    file_name = f"{sex}_chem_adj.csv"

    raw_path = DATA_PATH / "worm_wiring"
    raw_path = raw_path / file_name

    adj_df = load_adjacency(raw_path)
    node_ids = adj_df.index
    nodes = create_node_data(node_ids)
    # get rid of any nodes which don't have a side designation
    adj_df, nodes, removed_nonlateral = select_lateral_nodes(adj_df, nodes)
    # then ensure the network is fully connected
    adj_df, nodes, removed_lcc = ensure_connected(adj_df, nodes)
    # then remove any nodes whose partner got removed by that process
    adj_df, nodes, removed_partner_lcc = select_lateral_nodes(adj_df, nodes)
    # REPEAT in case this removal of partners causes disconnection
    adj_df, nodes, removed_lcc2 = ensure_connected(adj_df, nodes)
    adj_df, nodes, removed_partner_lcc2 = select_lateral_nodes(adj_df, nodes)

    adjplot(adj_df.values, plot_type="scattermap")

    g = nx.from_pandas_adjacency(adj_df, create_using=nx.DiGraph)
    nx.write_edgelist(
        g, OUT_PATH / f"{sex}_chem_edgelist.csv", delimiter=",", data=["weight"]
    )

    nodes.to_csv(OUT_PATH / f"{sex}_chem_nodes.csv")

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
