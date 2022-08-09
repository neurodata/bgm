import numpy as np
from graspologic.utils import largest_connected_component
import pandas as pd


def get_paired_inds(meta, check_in=True, pair_key="pair", pair_id_key="pair_id"):
    pair_meta = meta.copy()
    pair_meta["_inds"] = range(len(pair_meta))

    # remove any center neurons
    pair_meta = pair_meta[pair_meta["hemisphere"].isin(["L", "R"])]

    # remove any neurons for which the other in the pair is not in the metadata
    if check_in:
        pair_meta = pair_meta[pair_meta[pair_key].isin(pair_meta.index)]

    # remove any pairs for which there is only one neuron
    pair_group_size = pair_meta.groupby(pair_id_key).size()
    remove_pairs = pair_group_size[pair_group_size == 1].index
    pair_meta = pair_meta[~pair_meta[pair_id_key].isin(remove_pairs)]

    # make sure each pair is "valid" now
    assert pair_meta.groupby(pair_id_key).size().min() == 2
    assert pair_meta.groupby(pair_id_key).size().max() == 2

    # sort into pairs interleaved
    pair_meta.sort_values([pair_id_key, "hemisphere"], inplace=True)
    lp_inds = pair_meta[pair_meta["hemisphere"] == "L"]["_inds"]
    rp_inds = pair_meta[pair_meta["hemisphere"] == "R"]["_inds"]

    # double check that everything worked
    assert (
        meta.iloc[lp_inds][pair_id_key].values == meta.iloc[rp_inds][pair_id_key].values
    ).all()
    return lp_inds, rp_inds


def get_paired_subgraphs(adj, lp_inds, rp_inds):
    ll_adj = adj[np.ix_(lp_inds, lp_inds)]
    rr_adj = adj[np.ix_(rp_inds, rp_inds)]
    lr_adj = adj[np.ix_(lp_inds, rp_inds)]
    rl_adj = adj[np.ix_(rp_inds, lp_inds)]
    return (ll_adj, rr_adj, lr_adj, rl_adj)


def to_largest_connected_component(adj, meta=None):
    adj, lcc_inds = largest_connected_component(adj, return_inds=True)
    if meta is not None:
        return adj, meta.iloc[lcc_inds]
    else:
        return adj


def to_pandas_edgelist(g):
    """Works for multigraphs, the networkx one wasnt returning edge keys"""
    rows = []
    for u, v, k in g.edges(keys=True):
        data = g.edges[u, v, k]
        data["source"] = u
        data["target"] = v
        data["key"] = k
        rows.append(data)
    edges = pd.DataFrame(rows)
    edges["edge"] = list(zip(edges["source"], edges["target"], edges["key"]))
    edges.set_index("edge", inplace=True)
    return edges


def get_paired_nodes(nodes):
    paired_nodes = nodes[nodes["pair_id"] != -1]
    pair_ids = paired_nodes["pair_id"]

    pair_counts = pair_ids.value_counts()
    pair_counts = pair_counts[pair_counts == 1]
    pair_ids = pair_ids[pair_ids.isin(pair_counts.index)]

    paired_nodes = paired_nodes[paired_nodes["pair_id"].isin(pair_ids)].copy()

    return paired_nodes


def get_seeds(left_nodes, right_nodes):
    left_paired_nodes = get_paired_nodes(left_nodes)
    right_paired_nodes = get_paired_nodes(right_nodes)

    pairs_in_both = np.intersect1d(
        left_paired_nodes["pair_id"], right_paired_nodes["pair_id"]
    )
    left_paired_nodes = left_paired_nodes[
        left_paired_nodes["pair_id"].isin(pairs_in_both)
    ]
    right_paired_nodes = right_paired_nodes[
        right_paired_nodes["pair_id"].isin(pairs_in_both)
    ]

    left_seeds = left_paired_nodes.sort_values("pair_id")["inds"]
    right_seeds = right_paired_nodes.sort_values("pair_id")["inds"]

    assert (
        left_nodes.iloc[left_seeds]["pair_id"].values
        == right_nodes.iloc[right_seeds]["pair_id"].values
    ).all()

    return (left_seeds, right_seeds)


def remove_group(
    left_adj, right_adj, left_nodes, right_nodes, group, group_key="simple_group"
):
    left_nodes["inds"] = range(len(left_nodes))
    sub_left_nodes = left_nodes[left_nodes[group_key] != group]
    sub_left_inds = sub_left_nodes["inds"].values
    right_nodes["inds"] = range(len(right_nodes))
    sub_right_nodes = right_nodes[right_nodes[group_key] != group]
    sub_right_inds = sub_right_nodes["inds"].values

    sub_left_adj = left_adj[np.ix_(sub_left_inds, sub_left_inds)]
    sub_right_adj = right_adj[np.ix_(sub_right_inds, sub_right_inds)]

    return sub_left_adj, sub_right_adj, sub_left_nodes, sub_right_nodes


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


def create_node_data(node_ids, exceptions=[]):
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
        else:
            node_pair = node_id
            hemisphere = "C"

        node_rows.append(
            {"node_id": node_id, "pair": node_pair, "hemisphere": hemisphere}
        )

    nodes = pd.DataFrame(node_rows).set_index("node_id")

    return nodes


def get_hemisphere_indices(nodes):
    nodes = nodes.copy()
    nodes["_inds"] = np.arange(len(nodes))
    left_nodes = nodes[nodes["hemisphere"] == "L"]
    right_nodes = nodes[nodes["hemisphere"] == "R"]
    assert (left_nodes["pair"].values == right_nodes["pair"].values).all()
    left_indices = left_nodes["_inds"].values
    right_indices = right_nodes["_inds"].values
    return left_indices, right_indices


def split_connectome(adj, nodes):
    left_inds, right_inds = get_hemisphere_indices(nodes)
    A = adj[left_inds][:, left_inds]
    B = adj[right_inds][:, right_inds]
    AB = adj[left_inds][:, right_inds]
    BA = adj[right_inds][:, left_inds]
    return A, B, AB, BA
