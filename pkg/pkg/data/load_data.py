from pathlib import Path

import networkx as nx
import pandas as pd


DATA_PATH = Path(__file__).parent.parent.parent.parent  # don't judge me judge judy
DATA_PATH = DATA_PATH / "data"


def load_split_connectome(dataset, weights=True):
    dir = DATA_PATH / "processed_split"
    if weights:
        data = (("weight", int),)
    else:
        data = False
    if dataset in [
        "herm_chem",
        "male_chem",
        "specimen_107",
        "specimen_148",
    ]:
        nodetype = str
    elif dataset in ["maggot", "maggot_subset"]:
        nodetype = int
    g = nx.read_edgelist(
        dir / f"{dataset}_edgelist.csv",
        create_using=nx.DiGraph,
        delimiter=",",
        nodetype=nodetype,
        data=data,
    )
    nodes = pd.read_csv(dir / f"{dataset}_nodes.csv", index_col=0)
    adj = nx.to_numpy_array(g, nodelist=nodes.index)
    return adj, nodes


def load_semipaired_connectome(dataset, weights=True):
    loc = DATA_PATH / "processed_full"
    if weights:
        data = (("weight", int),)
    else:
        data = False
    if dataset in [
        "herm_chem",
        "male_chem",
        "specimen_107",
        "specimen_148",
    ]:
        nodetype = str
    elif dataset in ["maggot", "maggot_subset"]:
        nodetype = int
    g = nx.read_edgelist(
        loc / f"{dataset}_edgelist.csv",
        create_using=nx.DiGraph,
        delimiter=",",
        nodetype=nodetype,
        data=data,
    )
    nodes = pd.read_csv(loc / f"{dataset}_nodes.csv", index_col=0)
    adj = nx.to_numpy_array(g, nodelist=nodes.index)
    return adj, nodes
