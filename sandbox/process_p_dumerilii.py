#%%
import datetime
import time

import networkx as nx
import numpy as np
import pandas as pd
from graspologic.plot import adjplot
from pkg.data import DATA_PATH
from pkg.utils import create_node_data, ensure_connected, select_lateral_nodes


OUT_PATH = DATA_PATH / "processed_split"

t0 = time.time()


#%%


def load_adjacency(path, delimiter=","):
    adj_df = pd.read_csv(path, index_col=0, delimiter=delimiter).fillna(0)
    node_ids = np.union1d(adj_df.index, adj_df.columns)
    adj_df = adj_df.reindex(index=node_ids, columns=node_ids).fillna(0)
    adj_df = pd.DataFrame(
        data=adj_df.values.astype(int), index=adj_df.index, columns=adj_df.columns
    )
    return adj_df


#%%

#%%


def create_node_data(node_ids):
    rows = []
    for node_id in node_ids:
        chunks = node_id.split(" ")
        name = chunks[0]
        number = chunks[1].strip("#")

        n_l = name.count("l")
        n_r = name.count("r")
        is_pred_right = bool(np.argmax((n_l, n_r)))
        if is_pred_right:
            ind = name.rfind("r")
        else:
            ind = name.rfind("l")
        name_no_side = name[:ind] + name[ind + 1 :]

        rows.append(
            {
                "name": name,
                "id": number,
                "n_l": n_l,
                "n_r": n_r,
                "is_pred_right": is_pred_right,
                "pair": name_no_side,
            }
        )

    return pd.DataFrame(rows).set_index("name")


file_name = "randel_2014_eye_adjacency.csv"
raw_path = DATA_PATH / "p_dumerilii"
raw_path = raw_path / file_name
adj_df = load_adjacency(raw_path)

node_ids = adj_df.index.values
nodes = create_node_data(node_ids)
adj_df = pd.DataFrame(
    data=adj_df.values.astype(int), index=nodes.index, columns=nodes.index
)

pair_counts = nodes.groupby("pair").size()
pair_nodes = nodes[
    nodes["pair"].isin(pair_counts[pair_counts == 2].index.values)
].copy()
pair_nodes["hemisphere"] = "L"
pair_nodes.loc[pair_nodes["is_pred_right"], "hemisphere"] = "R"
pair_nodes.sort_values(["hemisphere", "pair"], inplace=True)
nodes = pair_nodes
adj_df = adj_df.reindex(index=nodes.index, columns=nodes.index)

from pkg.utils import select_lateral_nodes, ensure_connected

# get rid of any nodes which don't have a side designation
adj_df, nodes, removed_nonlateral = select_lateral_nodes(adj_df, nodes)
# then ensure the network is fully connected
adj_df, nodes, removed_lcc = ensure_connected(adj_df, nodes)
# then remove any nodes whose partner got removed by that process
adj_df, nodes, removed_partner_lcc = select_lateral_nodes(adj_df, nodes)
# REPEAT in case this removal of partners causes disconnection
adj_df, nodes, removed_lcc2 = ensure_connected(adj_df, nodes)
adj_df, nodes, removed_partner_lcc2 = select_lateral_nodes(adj_df, nodes)

g = nx.from_pandas_adjacency(adj_df, create_using=nx.DiGraph)

name = "annelid_eye_2"
nx.write_edgelist(g, OUT_PATH / f"{name}_edgelist.csv", delimiter=",", data=["weight"])

nodes.to_csv(OUT_PATH / f"{name}_nodes.csv")


#%%

file_name = "elife-02730-fig4-data1-v2.csv"
raw_path = DATA_PATH / "p_dumerilii"
raw_path = raw_path / file_name
raw_path = DATA_PATH / "p_dumerilii"
raw_path = raw_path / file_name
adj_df = load_adjacency(raw_path, delimiter=";")


def create_node_data(node_ids):
    rows = []
    for node_id in node_ids:
        # chunks = node_id.split(" ")
        name = node_id
        # number = chunks[1].strip("#")

        n_l = name.count("l")
        n_r = name.count("r")
        is_pred_right = bool(np.argmax((n_l, n_r)))
        if is_pred_right:
            ind = name.rfind("r")
        else:
            ind = name.rfind("l")
        name_no_side = name[:ind] + name[ind + 1 :]

        rows.append(
            {
                "name": name,
                # "id": number,
                "n_l": n_l,
                "n_r": n_r,
                "is_pred_right": is_pred_right,
                "pair": name_no_side,
            }
        )

    return pd.DataFrame(rows).set_index("name")


node_ids = adj_df.index.values
nodes = create_node_data(node_ids)
adj_df = pd.DataFrame(
    data=adj_df.values.astype(int), index=nodes.index, columns=nodes.index
)

pair_counts = nodes.groupby("pair").size()
pair_nodes = nodes[
    nodes["pair"].isin(pair_counts[pair_counts == 2].index.values)
].copy()
pair_nodes["hemisphere"] = "L"
pair_nodes.loc[pair_nodes["is_pred_right"], "hemisphere"] = "R"
pair_nodes.sort_values(["hemisphere", "pair"], inplace=True)
nodes = pair_nodes
adj_df = adj_df.reindex(index=nodes.index, columns=nodes.index)

from pkg.utils import select_lateral_nodes, ensure_connected

# get rid of any nodes which don't have a side designation
adj_df, nodes, removed_nonlateral = select_lateral_nodes(adj_df, nodes)
# then ensure the network is fully connected
adj_df, nodes, removed_lcc = ensure_connected(adj_df, nodes)
# then remove any nodes whose partner got removed by that process
adj_df, nodes, removed_partner_lcc = select_lateral_nodes(adj_df, nodes)
# REPEAT in case this removal of partners causes disconnection
adj_df, nodes, removed_lcc2 = ensure_connected(adj_df, nodes)
adj_df, nodes, removed_partner_lcc2 = select_lateral_nodes(adj_df, nodes)

g = nx.from_pandas_adjacency(adj_df, create_using=nx.DiGraph)


name = "annelid_visual"
nx.write_edgelist(g, OUT_PATH / f"{name}_edgelist.csv", delimiter=",", data=["weight"])

nodes.to_csv(OUT_PATH / f"{name}_nodes.csv")


#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
