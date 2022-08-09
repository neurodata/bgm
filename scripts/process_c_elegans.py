#%% [markdown]
# # *C. elegans* connectomes
#%%
import datetime
import time

import networkx as nx
import numpy as np
import pandas as pd
from graspologic.plot import adjplot
from pkg.data import DATA_PATH
from pkg.utils import create_node_data, ensure_connected, select_lateral_nodes


FILENAME = "process_c_elegans"

DISPLAY_FIGS = True


t0 = time.time()


#%% [markdown]
# ## Load the raw adjacency matrices
#%%


def load_adjacency(path):
    adj_df = pd.read_csv(path, index_col=0, header=0).fillna(0)
    node_ids = np.union1d(adj_df.index, adj_df.columns)
    adj_df = adj_df.reindex(index=node_ids, columns=node_ids).fillna(0)
    adj_df = pd.DataFrame(
        data=adj_df.values.astype(int), index=adj_df.index, columns=adj_df.columns
    )
    return adj_df


#%% [markdown]
# ## Filter data
# Make sure neurons are lateralized and fully connected
#%%
for sex in ["male", "herm"]:
    OUT_PATH = DATA_PATH / "processed_split"
    chem_name = f"{sex}_chem_adj.csv"

    ################
    # just get the chemical connections for simple initial experiments
    raw_path = DATA_PATH / "c_elegans"
    chem_path = raw_path / chem_name

    chem_df = load_adjacency(chem_path)

    # get some node information
    chem_node_ids = chem_df.index
    chem_nodes = create_node_data(chem_node_ids, exceptions=["vBWM", "dgl", "dBWM"])

    # get rid of any nodes which don't have a side designation
    chem_df, chem_nodes, removed_nonlateral = select_lateral_nodes(chem_df, chem_nodes)
    # then ensure the network is fully connected
    chem_df, chem_nodes, removed_lcc = ensure_connected(chem_df, chem_nodes)
    # then remove any nodes whose partner got removed by that process
    chem_df, chem_nodes, removed_partner_lcc = select_lateral_nodes(chem_df, chem_nodes)
    # REPEAT in case this removal of partners causes disconnection
    chem_df, chem_nodes, removed_lcc2 = ensure_connected(chem_df, chem_nodes)
    chem_df, chem_nodes, removed_partner_lcc2 = select_lateral_nodes(
        chem_df, chem_nodes
    )

    adjplot(chem_df.values, plot_type="scattermap")

    g = nx.from_pandas_adjacency(chem_df, create_using=nx.DiGraph)
    nx.write_edgelist(
        g, OUT_PATH / f"{sex}_chem_edgelist.csv", delimiter=",", data=["weight"]
    )

    chem_nodes.to_csv(OUT_PATH / f"{sex}_chem_nodes.csv")

    #################
    # now add the electrical
    # get some node information
    elec_name = f"{sex}_elec_adj.csv"
    elec_path = raw_path / elec_name

    elec_df = load_adjacency(elec_path)  # TODO: assert symmetric?
    chem_df = load_adjacency(chem_path)
    elec_node_ids = elec_df.index
    chem_node_ids = chem_df.index

    node_ids = np.union1d(elec_node_ids, chem_node_ids)
    elec_df = elec_df.reindex(index=node_ids, columns=node_ids).fillna(0)
    chem_df = chem_df.reindex(index=node_ids, columns=node_ids).fillna(0)
    sum_df = elec_df + chem_df

    nodes = create_node_data(node_ids, exceptions=["vBWM", "dgl", "dBWM"])

    # get rid of any nodes which don't have a side designation
    sum_side_df, side_nodes, removed_nonlateral = select_lateral_nodes(sum_df, nodes)
    # then ensure the network is fully connected
    sum_side_df, side_nodes, removed_lcc = ensure_connected(sum_side_df, side_nodes)
    # then remove any nodes whose partner got removed by that process
    sum_side_df, side_nodes, removed_partner_lcc = select_lateral_nodes(
        sum_side_df, side_nodes
    )
    # REPEAT in case this removal of partners causes disconnection
    sum_side_df, side_nodes, removed_lcc2 = ensure_connected(sum_side_df, side_nodes)
    sum_side_df, side_nodes, removed_partner_lcc2 = select_lateral_nodes(
        sum_side_df, side_nodes
    )

    side_node_ids = side_nodes.index
    side_elec_df = elec_df.reindex(index=side_node_ids, columns=side_node_ids).astype(
        int
    )
    side_chem_df = chem_df.reindex(index=side_node_ids, columns=side_node_ids).astype(
        int
    )
    elec_g = nx.from_pandas_adjacency(side_elec_df, create_using=nx.Graph)
    chem_g = nx.from_pandas_adjacency(side_chem_df, create_using=nx.DiGraph)

    OUT_PATH = DATA_PATH / "processed_full"

    nx.write_edgelist(
        chem_g, OUT_PATH / f"{sex}_chem_edgelist.csv", delimiter=",", data=["weight"]
    )
    nx.write_edgelist(
        elec_g, OUT_PATH / f"{sex}_elec_edgelist.csv", delimiter=",", data=["weight"]
    )

    side_nodes.to_csv(OUT_PATH / f"{sex}_nodes.csv")


#%% [markdown]
# ## End
#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
