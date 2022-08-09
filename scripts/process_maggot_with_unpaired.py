#%% [markdown]
# # Maggot connectome subset
#%%
import datetime
import logging
import time

import networkx as nx
import numpy as np
import pandas as pd
import pymaid
from pkg.data import DATA_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.plot import (
    dashes,
    matched_stripplot,
    method_palette,
    rgb2hex,
    set_theme,
    simple_plot_neurons,
    subgraph_palette,
)
from pkg.utils import ensure_connected

FILENAME = "process_maggot_with_unpaired"

DISPLAY_FIGS = True

OUT_PATH = DATA_PATH / "processed_full"


def glue(name, var, **kwargs):
    default_glue(name, var, FILENAME, **kwargs)


t0 = time.time()
set_theme()
rng = np.random.default_rng(8888)


pymaid.CatmaidInstance("https://l1em.catmaid.virtualflybrain.org/", None)
logging.getLogger("pymaid").setLevel(logging.WARNING)
pymaid.clear_cache()


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
    if annot_name == "Imambocus et al":
        indicator.name = "Imambocus et al. 2022"
    series_ids.append(indicator)
nodes = pd.concat(series_ids, axis=1, ignore_index=False).fillna(False)

#%%

# TODO: currently it is necessary to grab this information locally
raw_path = DATA_PATH / "maggot"
paired_nodes = pd.read_csv(raw_path / "nodes.csv", index_col=0)

#%%

temp_meta = pd.read_csv("bgm/data/maggot/meta_data.csv", index_col=0)
temp_meta = temp_meta[temp_meta["hemisphere"] != "C"]

#%%
intersect_ids = nodes.index.intersection(temp_meta.index)
nodes = nodes.loc[intersect_ids]
nodes["hemisphere"] = temp_meta.loc[intersect_ids, "hemisphere"]

#%%
nodes["pair"] = np.nan
nodes.loc[paired_nodes.index, "pair"] = paired_nodes["pair"]
nodes["pair"] = nodes["pair"].astype("Int64")
nodes

#%%
nodes = nodes.sort_values("hemisphere")
nodes


#%%

adj_df = pymaid.adjacency_matrix(nodes.index.values)
adj_df = pd.DataFrame(
    data=adj_df.values.astype(int), index=adj_df.index, columns=adj_df.columns
)

#%%

adj_df, nodes, removed_lcc = ensure_connected(adj_df, nodes)

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
