Search.setIndex({"docnames": ["abstract", "connectome_seeded", "connectomes", "explain", "introduction", "landing", "process_c_elegans", "process_maggot", "process_p_pacificus", "references", "simulations", "slides/cambridge/cambridge", "slides/nmc2022/nmc2022", "slides/talk/talk"], "filenames": ["abstract.md", "connectome_seeded.ipynb", "connectomes.ipynb", "explain.ipynb", "introduction.md", "landing.md", "process_c_elegans.ipynb", "process_maggot.ipynb", "process_p_pacificus.ipynb", "references.md", "simulations.ipynb", "slides/cambridge/cambridge.md", "slides/nmc2022/nmc2022.md", "slides/talk/talk.md"], "titles": ["Abstract", "Maggot connectome subset with seeds", "Connectome data", "Explain GM vs. BGM", "Introduction", "Welcome", "<em>C. elegans</em> connectomes", "Maggot connectome subset", "<em>P. pacificus</em> pharynx connectomes", "References", "Simulation", "Graph matching for connectomics", "Bisected graph matching", "Graph matching for connectomics"], "terms": {"graph": [0, 1, 3, 10], "match": [0, 1, 3, 10], "algorithm": [0, 10, 11, 12, 13], "attempt": 0, "find": [0, 12], "best": 0, "correspond": [0, 11], "between": [0, 10, 12, 13], "node": [0, 1, 2, 3, 6, 8, 10, 13], "two": [0, 10, 11, 13], "network": [0, 5, 6, 8, 10], "These": 0, "techniqu": 0, "have": [0, 1, 6, 8, 11, 13], "previous": [0, 7], "been": 0, "us": [0, 5, 10, 12], "individu": 0, "neuron": [0, 1, 5, 6, 8, 12], "nanoscal": 0, "connectom": [0, 5], "particular": 0, "pair": [0, 1, 3, 12], "across": [0, 1], "hemispher": [0, 1, 7, 10, 11, 12, 13], "howev": 0, "sinc": 0, "deal": 0, "specif": 0, "thei": [0, 11], "onli": [0, 1, 2, 10], "util": [0, 2, 6, 7, 8], "ipsilater": [0, 3, 10, 11, 13], "same": [0, 10, 11], "subgraph": [0, 10, 11, 13], "when": 0, "perform": 0, "here": [0, 11], "we": [0, 1, 10, 12], "present": 0, "modif": [0, 13], "state": 0, "art": 0, "which": [0, 1, 2, 6, 8, 13], "allow": [0, 11, 13], "solv": [0, 2, 10, 12, 13], "what": 0, "call": [0, 11], "bisect": [0, 3, 10], "problem": [0, 13], "thi": [0, 1, 5, 6, 7, 8], "connect": [0, 6, 8], "brain": [0, 12], "predict": 0, "show": [0, 10, 13], "simul": [0, 3], "well": 0, "real": 0, "exampl": [0, 11, 12, 13], "edg": [0, 7, 10], "correl": [0, 10, 11, 13], "contralater": [0, 2, 3, 7, 10], "approach": [0, 12], "improv": [0, 2, 12, 13], "accuraci": [0, 1, 10, 11, 13], "expect": 0, "our": 0, "propos": [0, 13], "method": [0, 1, 2, 3, 10, 11, 13], "futur": 0, "endeavor": 0, "accur": 0, "other": [0, 11], "applic": 0, "where": [0, 10], "aris": 0, "import": [1, 2, 3, 6, 7, 8, 10, 11], "datetim": [1, 2, 3, 6, 7, 8, 10], "time": [1, 2, 3, 6, 8, 10], "matplotlib": [1, 2, 3, 7, 10], "pyplot": [1, 2, 3, 7, 10], "plt": [1, 2, 3, 7, 10], "numpi": [1, 2, 3, 6, 7, 10], "np": [1, 2, 3, 6, 7, 10], "panda": [1, 2, 6, 7, 8, 10], "pd": [1, 2, 6, 7, 8, 10], "seaborn": [1, 2, 3, 7, 10], "sn": [1, 2, 3, 7, 10], "from": [1, 2, 3, 6, 7, 8, 10], "graspolog": [1, 3, 6, 7, 8, 10, 12, 13], "graph_match": [1, 11, 13], "line": [1, 3], "line2d": 1, "transform": [1, 3], "blended_transform_factori": 1, "pkg": [1, 2, 3, 6, 7, 8, 10], "data": [1, 3, 10, 12], "load_semipaired_connectom": 1, "io": [1, 2, 3, 7, 10, 11, 13], "out_path": [1, 2, 3, 6, 7, 8, 10], "glue": [1, 2, 3, 7, 10], "default_glu": [1, 2, 3, 7, 10], "savefig": [1, 2, 3, 7, 10], "plot": [1, 3, 6, 8, 10], "dash": [1, 10], "method_palett": [1, 2, 10], "rgb2hex": 1, "set_them": [1, 2, 3, 7, 10], "simple_plot_neuron": 1, "subgraph_palett": 1, "scipi": [1, 2, 13], "optim": [1, 11, 13], "linear_sum_assign": 1, "stat": [1, 2], "wilcoxon": [1, 2], "sklearn": 1, "model_select": 1, "kfold": 1, "tqdm": [1, 2, 10], "autonotebook": 1, "filenam": [1, 2, 3, 6, 7, 10], "connectome_seed": 1, "display_fig": [1, 2, 3, 6, 7, 10], "true": [1, 2, 3, 6, 7, 10, 11], "def": [1, 2, 3, 6, 7, 10], "name": [1, 2, 3, 7, 10], "var": [1, 2, 3, 7, 10], "kwarg": [1, 2, 3, 7, 10], "gluefig": [1, 2, 3, 7, 10], "fig": [1, 2, 3, 7, 10], "foldernam": [1, 2, 3, 7, 10], "figur": [1, 2, 3, 7, 10], "close": [1, 2, 3, 7, 10], "t0": [1, 2, 3, 6, 7, 8, 10], "rng": [1, 2, 3, 7, 10], "random": [1, 3, 7, 10], "default_rng": [1, 2, 3, 7, 10], "8888": [1, 2, 3, 7, 10], "adj": [1, 2, 3], "maggot_subset": [1, 2, 3], "adj_df": [1, 6, 7, 8], "datafram": [1, 2, 6, 7, 8, 10], "index": [1, 2, 3, 6, 7, 8], "column": [1, 2, 6, 7, 8, 11, 13], "left_nod": 1, "l": [1, 7], "right_nod": 1, "r": [1, 3, 7], "left_node_id": 1, "right_node_id": 1, "ll_adj": 1, "reindex": [1, 6, 7, 8], "valu": [1, 2, 3, 6, 7, 8, 10], "rr_adj": 1, "lr_adj": 1, "rl_adj": 1, "n_left": 1, "len": [1, 2, 3, 7], "n_right": 1, "select_se": 1, "all": [1, 12, 13], "copi": [1, 3], "node_id": [1, 6, 7], "pos_index": 1, "rang": [1, 10], "left_paired_nod": 1, "isna": 1, "right_paired_nod": 1, "intersect1d": 1, "reset_index": 1, "set_index": [1, 7], "left_ind": [1, 2], "loc": [1, 2, 7, 10], "right_ind": [1, 2], "column_stack": 1, "return": [1, 2, 3, 6, 7], "all_se": 1, "indic": [1, 7, 11], "arang": [1, 2], "n_seeds_rang": 1, "0": [1, 2, 3, 6, 7, 8, 10, 11, 13], "100": 1, "200": 1, "300": 1, "400": 1, "n_fold": 1, "10": [1, 2, 3, 6, 7, 8, 10, 11, 13], "rerun": [1, 7], "row": [1, 2, 3, 7, 10, 11, 13], "n_split": 1, "shuffl": 1, "random_st": 1, "integ": [1, 2], "iinfo": [1, 2], "uint32": [1, 2], "max": [1, 2], "pbar": 1, "total": 1, "2": [1, 2, 3, 7, 10], "fold": 1, "indices_train": 1, "indices_test": 1, "enumer": [1, 2, 3], "split": [1, 7], "test_se": 1, "left_nodes_to_check": 1, "iloc": 1, "n_seed": 1, "selected_se": 1, "gm": [1, 2, 10, 11, 13], "bgm": [1, 2, 7, 10, 11, 13], "ab": [1, 2, 3, 10, 11], "none": [1, 3, 7], "ba": [1, 2, 3, 10, 11], "els": [1, 2, 3, 7], "indices_a": [1, 11], "indices_b": [1, 11], "score": [1, 2, 11], "misc": [1, 11], "n_init": [1, 11], "1": [1, 2, 3, 7, 10], "partial_match": [1, 11], "left_nodes_sort": 1, "right_nodes_sort": 1, "correct": [1, 11, 13], "get_loc": 1, "check": 1, "isinst": 1, "bool_": 1, "match_ratio_heldout": 1, "append": [1, 2, 7, 10], "match_ratio": [1, 2, 10], "updat": 1, "result": [1, 2, 11], "to_csv": [1, 2, 6, 7, 8], "matching_result": 1, "csv": [1, 2, 6, 7, 8], "read_csv": [1, 2, 6, 7], "index_col": [1, 2, 6, 7], "ax": [1, 2, 3, 7, 10], "subplot": [1, 2, 3, 7, 10], "figsiz": [1, 2, 3, 7, 10], "8": [1, 3, 10, 11, 13], "6": [1, 2, 3, 7, 10], "lineplot": [1, 10], "x": [1, 2, 3, 7, 10, 11, 13], "y": [1, 2, 3, 7, 10], "hue": [1, 2, 10], "palett": [1, 2, 3, 10], "style": [1, 10], "move_legend": [1, 10], "lower": 1, "right": [1, 2, 3, 7, 10, 11, 13], "titl": [1, 2, 3, 10], "set": [1, 2, 3, 7, 10, 13], "ylabel": [1, 2, 7], "xlabel": [1, 2], "number": [1, 10, 11, 13], "xtick": [1, 3], "accuracy_by_se": 1, "stat_row": 1, "seed_result": 1, "groupbi": [1, 2, 3, 10], "ratio": [1, 2], "method_result": 1, "believ": [1, 11, 13], "should": 1, "pvalu": [1, 2], "mannwhitneyu": 1, "stat_result": 1, "seeded_stat_result": 1, "full_seed_n_init": 1, "9999": 1, "match_prob": 1, "zero": [1, 3], "shape": 1, "i": [1, 2, 3, 11], "match_probs_df": 1, "full": [1, 10, 13], "probabl": [1, 7], "get": [1, 6, 8], "most": [1, 13], "like": 1, "each": [1, 10, 11], "maxim": [1, 11], "ar": [1, 6, 7, 8], "final": [1, 11], "choos": 1, "one": [1, 2, 3, 10, 13], "p_match": 1, "resort": 1, "accordingli": 1, "saniti": 1, "alwai": [1, 11], "equal_pair": 1, "real_pair": 1, "mean": [1, 2, 10], "print": [1, 2, 3, 6, 7, 8, 10], "f": [1, 2, 3, 6, 7, 8, 10, 11, 13], "sub": 1, "select": [1, 11], "new": [1, 13], "new_pair": 1, "make": [1, 6, 7, 8, 11], "new_left_nod": 1, "skid_left": 1, "new_right_nod": 1, "skid_right": 1, "pair_df": 1, "concat": [1, 2, 7], "to_seri": 1, "drop": 1, "axi": [1, 2, 3, 7, 11, 13], "remov": [1, 3, 6, 7, 8], "anyth": 1, "never": 1, "actual": [1, 7], "got": [1, 6, 7, 8], "equal": [1, 3], "size": [1, 2, 3, 11], "distribut": 1, "histplot": 1, "draw_box": 1, "color": [1, 2, 3], "black": [1, 3], "ref": 1, "http": [1, 3, 7, 13], "github": [1, 11, 12, 13], "com": [1, 12], "blob": 1, "81e955935a26dae7048758f7b3dc3f1dc4c5de6c": 1, "lib": 1, "_ax": 1, "py": 1, "l749": 1, "xtran": 1, "get_xaxis_transform": 1, "grid": 1, "ytran": 1, "get_yaxis_transform": 1, "xmin": 1, "095": 1, "xmax": 1, "09": [1, 11], "ymin": 1, "ymax": 1, "tran": 1, "dict": [1, 2, 3, 11], "clip_on": [1, 2], "fals": [1, 2, 3, 7, 10], "add_lin": 1, "plot_paired_neuron": 1, "left_id": 1, "right_id": 1, "n_show": 1, "n_col": [1, 2], "n_row": [1, 2], "3": [1, 2, 3, 10], "view": 1, "elev": 1, "90": 1, "azim": 1, "45": [1, 2, 7, 8], "constrained_layout": [1, 2], "gs": 1, "gridspec": 1, "hspace": [1, 2, 3], "wspace": [1, 3], "ll": [1, 3, 10, 11, 13], "rr": [1, 3, 10, 11, 13], "empti": 1, "dtype": [1, 7], "object": [1, 7, 11, 13], "j": [1, 11], "zip": [1, 3], "add_subplot": 1, "project": [1, 11], "3d": 1, "force_bound": 1, "autoscal": 1, "soma": 1, "dist": 1, "lw": 1, "5": [1, 2, 3, 11, 13], "lightgrei": [1, 3], "morpholog": [1, 13], "some": [1, 11], "good": 1, "7": [1, 2, 3, 11], "best_pair_df": 1, "sampl": [1, 10], "n": [1, 2, 3, 10, 11], "replac": [1, 7], "example_matched_morphologies_good": 1, "bad": 1, "worst_pair_df": 1, "sort_valu": [1, 2, 7], "example_matched_morphologies_bad": 1, "elaps": [1, 2, 3, 6, 7, 8, 10], "delta": [1, 2, 3, 6, 7, 8, 10], "timedelta": [1, 2, 3, 6, 7, 8, 10], "second": [1, 2, 3, 6, 7, 8, 10], "script": [1, 2, 3, 6, 7, 8, 10], "took": [1, 2, 3, 6, 7, 8, 10], "complet": [1, 2, 3, 6, 7, 8, 10, 13], "now": [1, 2, 3, 6, 7, 8, 10], "mpl": 2, "load_split_connectom": [2, 3], "graphmatchsolv": [2, 10], "matched_stripplot": 2, "get_hemisphere_indic": 2, "compute_contralateral_ratio": 2, "A": [2, 3, 10], "b": [2, 3, 10], "agg": 2, "nonzero": 2, "aggfunc": 2, "count_nonzero": [2, 7], "elif": [2, 3, 10], "sum": [2, 7, 11, 13], "m_a": 2, "m_b": 2, "m_ab": 2, "m_ba": 2, "rerun_sim": 2, "dataset": 2, "male_chem": 2, "herm_chem": 2, "specimen_148": 2, "specimen_107": 2, "n_sim": [2, 10], "50": 2, "n_initi": 2, "contra_weight_ratio": 2, "results_by_dataset": 2, "n_node": 2, "_n_node": 2, "form": [2, 7, 10], "long": [2, 10, 11], "n_edg": 2, "_n_edg": 2, "contra_edge_ratio": 2, "_contra_edge_ratio": 2, "0f": [2, 7, 10], "_contra_weight_ratio": 2, "n_side": [2, 10], "sim": [2, 10], "leav": [2, 10], "solver": [2, 10], "run_start": 2, "permutation_": [2, 10], "converg": 2, "n_iter": 2, "score_": 2, "_match_result": 2, "font_scal": [2, 3], "scale": [2, 3], "jitter": 2, "25": 2, "meanline_width": 2, "35": [2, 7, 11], "n_dataset": 2, "order": [2, 11, 13], "nice_dataset_map": 2, "c": 2, "elegan": 2, "nhermaphrodit": 2, "nmale": 2, "maggot": [2, 5], "d": 2, "melanogast": 2, "larva": 2, "subset": 2, "p": [2, 3, 10], "pacificu": 2, "npharynx": 2, "int": [2, 3, 6, 7, 8], "ceil": 2, "min": [2, 7], "sharei": 2, "gridspec_kw": [2, 3], "acc_chang": 2, "item": 2, "unravel_index": 2, "legend": 2, "tick_param": 2, "both": [2, 10, 11, 13], "length": 2, "set_ylabel": [2, 10], "set_xlabel": [2, 10], "set_titl": 2, "ticklabel": 2, "get_xticklabel": [2, 7], "get_text": 2, "set_color": [2, 3], "gm_result": 2, "bgm_result": 2, "mode": 2, "approx": [2, 11], "_match_ratio_pvalu": 2, "_mean_accuracy_chang": 2, "mean_match_ratio": 2, "text": [2, 3], "05": [2, 3, 6, 7, 8, 10, 11, 13], "2f": [2, 7], "va": [2, 3, 7], "center": [2, 3, 7, 11], "ha": [2, 3, 7], "left": [2, 3, 10, 11, 13], "fontsiz": [2, 3], "medium": [2, 3], "_": [2, 3, 7], "_mean_match_accuraci": 2, "set_xlim": [2, 3], "set_ytick": 2, "75": 2, "9": [2, 3, 10], "yaxi": 2, "set_major_loc": 2, "maxnloc": 2, "4": [2, 3, 11], "flat": [2, 3], "has_data": 2, "off": [2, 3], "meta_result": 2, "weight": [2, 6, 7, 8, 11, 13], "t": [2, 3, 6, 8, 10, 12], "scatterplot": 2, "axessubplot": 2, "all_result": 2, "683871": 2, "200426005": 2, "397930": 2, "30": [2, 10, 11], "833871": 2, "004004": 2, "15": 2, "690323": 2, "1087836691": 2, "263822": 2, "854839": 2, "557582": 2, "600000": 2, "2808301671": 2, "705772": 2, "95": [2, 10], "000000": 2, "47": 2, "3948372264": 2, "000656": 2, "96": 2, "555556": 2, "48": 2, "1655188775": 2, "000773": 2, "97": 2, "000657": 2, "98": 2, "49": 2, "2605143165": 2, "000769": 2, "99": 2, "000672": 2, "500": 2, "rcparam": 2, "hatch": 2, "linewidth": [2, 3], "wa": [2, 10], "sort": 2, "seri": [2, 5, 7, 11, 13], "barplot": [2, 7], "hue_ord": [2, 10], "edgecolor": 2, "white": [2, 3], "zorder": [2, 3], "errcolor": 2, "errwidth": 2, "hack": [2, 8], "add": [2, 13], "leg": 2, "get_legend": 2, "handl": 2, "label": [2, 3], "get_legend_handles_label": 2, "upper": [2, 10], "bbox_to_anchor": 2, "frameon": [2, 10], "setp": [2, 7], "rotat": [2, 7], "top": [2, 3], "rotation_mod": [2, 7], "anchor": [2, 7], "set_xticklabel": 2, "map": [2, 11, 13], "draw_signific": 2, "xdist": 2, "02": 2, "ydist": 2, "03": 2, "0005": 2, "005": 2, "dimgrei": 2, "bottom": 2, "larg": [2, 3, 11], "set_ylim": [2, 3], "get_ylim": [2, 3], "xaxi": 2, "set_label_coord": 2, "17": [2, 11], "match_accuracy_comparison": 2, "06": [2, 7], "04": [2, 7], "185500": 2, "2022": [2, 3, 6, 7, 8, 10, 11, 13], "23": [2, 3, 6, 7, 8, 10], "11": [2, 7, 10], "39": 2, "665421": 2, "log": [3, 7], "navi": 3, "pymaid": [3, 7], "heatmap": [3, 8], "er_corr": [3, 10], "patch": 3, "rectangl": 3, "patheffect": 3, "normal": 3, "stroke": 3, "merge_ax": 3, "multicolor_text": 3, "seed": [3, 10, 13], "888": 3, "catmaidinst": [3, 7], "l1em": [3, 7], "catmaid": 3, "virtualflybrain": [3, 7], "org": [3, 7, 11, 13], "getlogg": [3, 7], "setlevel": [3, 7], "warn": [3, 7], "clear_cach": [3, 7], "info": [3, 7], "global": [3, 7], "instanc": 3, "cach": [3, 7], "ON": [3, 7], "rescal": 3, "orient": 3, "lim": 3, "get_xlim": 3, "extent": 3, "boost": 3, "new_lim": 3, "color_palett": 3, "set2": 3, "n_pair": 3, "show_pair": 3, "choic": 3, "count": [3, 7, 8], "id": [3, 7], "nl": 3, "get_neuron": 3, "plot2d": 3, "2d": 3, "ytick": 3, "invert_xaxi": 3, "set_alpha": 3, "spine": 3, "set_vis": 3, "pad": [3, 11], "transax": 3, "set_facecolor": 3, "w": 3, "neuron_galleri": 3, "20": 3, "height_ratio": 3, "direct": [3, 10], "permut": [3, 10, 13], "side": [3, 6, 8, 10, 11, 13], "appropri": [3, 10], "perm": [3, 10, 11], "undo_perm": [3, 10], "argsort": [3, 10], "ns": 3, "subgraph_label": 3, "lr": [3, 10, 11, 13], "rl": [3, 10, 11, 13], "annotated_heatmap": 3, "label_ipsi": 3, "label_contra": 3, "label_sid": 3, "inner_hier_label": 3, "cbar": 3, "cmap": 3, "grei": 3, "hier_label_fonts": 3, "vmax": 3, "rect_kw": 3, "alpha": 3, "width": 3, "height": 3, "add_patch": 3, "nice_text": 3, "a_": [3, 10, 11, 13], "get_lin": 3, "line_obj": 3, "s": [3, 7, 13], "transdata": 3, "set_path_effect": 3, "foreground": 3, "draw_permut": 3, "bump": 3, "horizont": 3, "axhlin": 3, "target": 3, "sourc": 3, "sign": 3, "xy": 3, "xytext": 3, "vertic": 3, "annot": [3, 11], "arrowprop": 3, "arrowstyl": 3, "connectionstyl": 3, "arc3": 3, "rad": 3, "facecolor": 3, "draw_match": 3, "adjac": 3, "matrix": 3, "adj_no_contra": 3, "title_obj": 3, "set_x": 3, "get_titl": 3, "set_": 3, "adj_only_contra": 3, "40": 3, "nmatch": 3, "27": 3, "rightarrow": [3, 11], "space": [3, 11, 13], "min_p": 3, "_f": [3, 10], "col": 3, "025": 3, "00": [3, 6, 7, 8, 10], "461674": 3, "59": [3, 7], "43": 3, "725097": 3, "websit": 5, "host": 5, "paper": 5, "studi": 5, "larval": 5, "drosophila": 5, "The": [5, 10, 13], "gener": [5, 11, 13], "jupyt": [5, 11, 13], "book": [5, 11, 13], "render": 5, "notebook": 5, "layout": 5, "few": 5, "skeleton": 5, "drawn": 5, "border": 5, "networkx": [6, 7, 8], "nx": [6, 7, 8], "adjplot": [6, 7, 8], "data_path": [6, 7, 8], "create_node_data": [6, 8], "ensure_connect": [6, 7, 8], "select_lateral_nod": [6, 7, 8], "process_c_elegan": 6, "processed_split": [6, 7, 8], "load_adjac": 6, "path": [6, 8, 11], "fillna": [6, 7], "union1d": 6, "astyp": [6, 7, 8], "sure": [6, 7, 8], "later": [6, 7, 8], "fulli": [6, 7, 8], "sex": 6, "male": 6, "herm": 6, "file_nam": 6, "_chem_adj": 6, "raw_path": [6, 7], "c_elegan": 6, "except": 6, "vbwm": 6, "dgl": 6, "dbwm": 6, "rid": [6, 8], "ani": [6, 7, 8, 13], "don": [6, 8], "design": [6, 8], "removed_nonlater": [6, 7, 8], "ensur": [6, 7, 8], "removed_lcc": [6, 7, 8], "whose": [6, 7, 8], "partner": [6, 7, 8], "process": [6, 7, 8], "removed_partner_lcc": [6, 7, 8], "repeat": [6, 7, 8], "case": [6, 7, 8], "caus": [6, 7, 8], "disconnect": [6, 7, 8], "removed_lcc2": [6, 7, 8], "removed_partner_lcc2": [6, 7, 8], "plot_typ": [6, 7, 8], "scattermap": [6, 7], "g": [6, 7, 8], "from_pandas_adjac": [6, 7, 8], "create_us": [6, 7, 8], "digraph": [6, 7, 8], "write_edgelist": [6, 7, 8], "_chem_edgelist": 6, "delimit": [6, 7, 8], "_chem_nod": 6, "501076": 6, "58": [6, 8], "33": 6, "953015": 6, "process_maggot": 7, "get_indicator_from_annot": 7, "annot_nam": 7, "filt": 7, "get_skids_by_annot": 7, "ones": 7, "bool": 7, "annot_df": 7, "get_annot": 7, "series_id": 7, "imambocu": 7, "et": [7, 11, 12, 13], "al": [7, 11, 12, 13], "ignore_index": 7, "zwart": 7, "2016": 7, "berck": 7, "khandelw": 7, "eichler": 7, "li": 7, "litwin": 7, "kumar": 7, "2017": 7, "larderet": 7, "fritsch": 7, "ohyama": 7, "schneider": 7, "mizel": 7, "2015": [7, 11, 12], "jovan": 7, "schlegel": 7, "fushiki": 7, "takagi": 7, "heckscher": 7, "gerhard": 7, "burgo": 7, "2018": [7, 11], "miroschnikow": 7, "2019": [7, 11], "carreira": 7, "rosario": 7, "arzan": 7, "zarin": 7, "clark": 7, "mark": 7, "andrad": 7, "tastekin": 7, "eschbach": [7, 12], "2020": 7, "2020b": 7, "hueckesfeld": 7, "vald": 7, "aleman": 7, "2021": [7, 11, 12], "section": 7, "meant": 7, "forthcom": 7, "wind": [7, 12, 13], "pedigo": 7, "were": [7, 10], "temp": 7, "code": [7, 12, 13], "run": [7, 10], "again": 7, "pair_id_count": 7, "iterrow": 7, "leftid": 7, "rightid": 7, "value_count": 7, "duplic": 7, "bad_pair": 7, "isin": 7, "inplac": 7, "local": 7, "just": 7, "adjacency_matrix": 7, "left_index": 7, "right_index": 7, "m_ll": 7, "m_rr": 7, "m_lr": 7, "m_rl": 7, "m_contra": 7, "m_ipsi": 7, "p_contra": 7, "an": [7, 13], "being": 7, "maggot_subset_edgelist": 7, "maggot_subset_nod": 7, "annotations_year": 7, "sort_index": 7, "kind": 7, "stabl": 7, "list": [7, 8], "ascend": 7, "n_pub": 7, "multi_pub_annot": 7, "idx": 7, "4123145": 7, "11106522": 7, "6578062": 7, "12820178": 7, "2798040": 7, "4542822": 7, "10858401": 7, "12809976": 7, "4386719": 7, "6570401": 7, "16848475": 7, "17176866": 7, "9841469": 7, "11637003": 7, "17980792": 7, "17176882": 7, "first_loc": 7, "argmax": 7, "first_publish": 7, "used_pap": 7, "uniqu": 7, "setdiff1d": 7, "miss": 7, "12": 7, "913105": 7, "335200": 7, "specimen": 8, "107": 8, "148": 8, "p_pacificu": 8, "specimen_": 8, "_synapselist": 8, "mg": 8, "read_edgelist": 8, "multidigraph": 8, "synaps": [8, 11, 13], "to_pandas_adjac": 8, "_edgelist": 8, "_node": 8, "319799": 8, "400713": 8, "fig_path": 10, "88888888": 10, "1000": 10, "ipsi_rho": 10, "ipsi_p": 10, "contra_p": 10, "let": 10, "erdo": 10, "reyni": 10, "written": 10, "correr": 10, "rho": 10, "densiti": 10, "independ": 10, "also": 10, "rho_": 10, "contra": 10, "defin": 10, "begin": 10, "bmatrix": 10, "appli": [10, 11, 13], "i_n": 10, "p_": 10, "rand": 10, "thu": 10, "can": [10, 12], "write": [10, 13], "vari": [10, 11, 13], "For": 10, "accord": [10, 11], "abov": [10, 11], "try": 10, "recov": 10, "align": 10, "min_": 10, "default": [10, 11], "initi": [10, 11, 12], "barycent": [10, 11], "maximum": [10, 11], "frank": [10, 11, 13], "wolf": [10, 11, 13], "fw": 10, "iter": [10, 11], "stop": [10, 11], "toler": [10, 11], "norm": [10, 11, 13], "differ": [10, 11, 13], "solut": 10, "01": 10, "comput": [10, 11], "contra_rho": 10, "linspac": 10, "desc": 10, "str": 10, "below": 10, "function": [10, 11, 13], "strength": 10, "three": 10, "shade": 10, "region": 10, "confid": 10, "interv": 10, "zero_acc": 10, "zero_diff": 10, "point_9_acc": 10, "point_9_diff": 10, "match_ratio_by_contra_rho": 10, "26": 10, "052283": 10, "22": 10, "512574": 10, "he": [11, 13], "him": [11, 13], "neurodata": [11, 12, 13], "lab": [11, 13], "john": [11, 13], "hopkin": [11, 13], "univers": [11, 13], "dept": [11, 13], "biomed": [11, 13], "engin": [11, 13], "bpedigo": [11, 12, 13], "jhu": [11, 12, 13], "edu": [11, 12, 13], "bdpedigo": [11, 13], "bpedigod": [11, 13], "twitter": [11, 13], "exist": [11, 13], "understand": [11, 13], "stereotypi": [11, 13], "downstream": [11, 13], "analysi": [11, 13], "embed": [11, 13], "collaps": [11, 13], "complex": [11, 13], "sens": 11, "week": 11, "observ": 11, "phone": 11, "chang": 11, "But": 11, "noisi": 11, "version": 11, "still": [11, 13], "those": 11, "pa": 11, "ap": 11, "so": 11, "want": 11, "pap": 11, "search": [11, 13], "fix": 11, "measur": [11, 13], "disagr": [11, 13], "unweight": [11, 13], "absurdli": 11, "335": 11, "82": 11, "than": 11, "atom": 11, "convex": [11, 13], "p_1": [11, 13], "p_2": [11, 13], "notin": [11, 13], "h": 11, "550": 11, "imag": 11, "png": 11, "350": 11, "rank": 11, "panel": 11, "relax": [11, 13], "birkoff": [11, 13], "polytop": [11, 13], "doubli": [11, 13], "stochast": [11, 13], "element": [11, 13], "transport": [11, 13], "oppos": [11, 13], "assign": [11, 13], "soft": [11, 13], "minim": [11, 13], "first": [11, 13], "taylor": [11, 13], "over": [11, 13], "requir": [11, 13], "gradient": [11, 13], "nabla": [11, 13], "back": 11, "done": 11, "graphmatch": 11, "paramet": 11, "go": 11, "fit": 11, "perm_inds_": 11, "b_permut": 11, "quadratic_assign": [11, 13], "faq": [11, 12, 13], "option": 11, "3k": 11, "550k": [11, 13], "With": 11, "vanilla": 11, "80": [11, 13], "expert": 11, "knowledg": 11, "nblast": 11, "summari": 11, "base": 11, "compart": 11, "seeds_a": 11, "seeds_b": 11, "If": 11, "a1": 11, "a2": 11, "b1": 11, "b2": 11, "textcolor": [11, 13], "66c2a5": [11, 13], "fc8d62": [11, 13], "tp": [11, 13], "8da0cb": [11, 13], "e78ac3": [11, 13], "creat": [11, 13], "had": [11, 13], "_s": 11, "sum_": 11, "k": 11, "tr": 11, "give": 11, "within": 11, "layer": 11, "axo": 11, "axon": 11, "dendrit": 11, "etc": 11, "see": 11, "max_it": 11, "control": 11, "poor": 11, "take": 11, "too": 11, "increas": 11, "tol": 11, "init": 11, "think": 11, "uninform": 11, "restart": 11, "typic": 11, "determinist": 11, "a_ll": 11, "a_rr": 11, "a_lr": 11, "a_rl": 11, "alignemnt": 11, "least": 11, "dictionari": 11, "detail": 11, "dpmcsuss": 11, "igraphmatch": 11, "tutori": [11, 13], "cours": [11, 13], "html": [11, 13], "microsoft": [11, 12, 13], "latest": [11, 13], "doc": [11, 13], "repo": [11, 13], "abstract": [11, 13], "manuscript": [11, 12, 13], "biorxiv": [11, 12, 13], "content": [11, 13], "1101": [11, 13], "19": [11, 13], "492713": [11, 13], "wip": 11, "pull": 11, "960": 11, "burkard": 11, "2012": 11, "societi": 11, "industri": 11, "doi": 11, "1137": 11, "9781611972238": 11, "vogelstein": [11, 12], "fast": 11, "approxim": 11, "quadrat": 11, "program": 11, "plo": 11, "ONE": 11, "1371": 11, "journal": 11, "pone": 11, "0121002": 11, "fishkind": 11, "pattern": 11, "recognit": 11, "87": 11, "203": 11, "215": 11, "1016": 11, "patcog": 11, "014": 11, "pantazi": 11, "multiplex": 11, "filter": 11, "scienc": 11, "1007": 11, "s41109": 11, "022": 11, "00464": 11, "saad": [11, 13], "eldin": [11, 13], "via": 11, "arxiv": 11, "2111": 11, "05366": 11, "autom": 11, "big": [11, 13], "thank": [11, 13], "contributor": 11, "especi": 11, "ali": [11, 13], "work": 11, "bilater": 12, "adapt": 12, "help": [12, 13], "more": 12, "tinyurl": 12, "neuromatch": 12, "experi": 12, "contact": 12, "co": 12, "author": 12, "michael": [12, 13], "carei": 12, "e": 12, "prieb": 12, "joshua": 12, "fund": 12, "nsf": 12, "grfp": 12, "career": 12, "nih": 12, "research": 12, "collabor": 13, "marta": 13, "zlatic": 13, "albert": 13, "cardona": 13, "group": 13, "led": 13, "whole": 13, "singl": 13, "cell": 13, "insect": 13, "3000": 13, "reconstruct": 13, "In": 13, "prep": 13, "proofread": 13, "omnibu": 13, "matric": 13, "similar": 13, "76": 13, "lot": 13, "bell": 13, "whistl": 13, "type": 13, "modifi": 13, "incorpor": 13, "inform": 13, "amount": 13, "simpl": 13, "previou": 13, "tool": 13, "suffici": 13, "demonstr": 13, "inde": 13, "five": 13, "refer": 13, "www": 13}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"abstract": 0, "maggot": [1, 7, 13], "connectom": [1, 2, 6, 7, 8, 11, 12, 13], "subset": [1, 7], "seed": [1, 2, 11], "data": [2, 6, 7, 8], "load": [2, 6, 7], "process": 2, "run": 2, "match": [2, 11, 12, 13], "experi": [2, 10], "plot": [2, 7], "accuraci": [2, 12], "show": 2, "each": 2, "random": 2, "aggreg": 2, "end": [2, 3, 6, 7, 8, 10], "explain": 3, "gm": [3, 12], "vs": 3, "bgm": [3, 12], "introduct": 4, "welcom": 5, "c": 6, "elegan": 6, "raw": 6, "adjac": [6, 7, 11], "matric": [6, 11], "filter": [6, 7, 8], "start": 7, "catmaid": 7, "instanc": 7, "virtual": 7, "fly": 7, "brain": [7, 11, 13], "paper": 7, "meta": 7, "annot": 7, "get": 7, "pair": [7, 11, 13], "connect": [7, 11, 13], "result": [7, 10], "matrix": [7, 11], "comput": 7, "some": 7, "simpl": 7, "statist": 7, "save": 7, "final": 7, "network": [7, 11, 12, 13], "node": [7, 11], "metadata": 7, "examin": [7, 11], "neuron": [7, 11, 13], "us": [7, 11, 13], "here": 7, "exampl": 7, "publish": 7, "twice": 7, "same": 7, "year": 7, "all": [7, 11], "which": [7, 11], "one": [7, 11], "our": 7, "wa": 7, "first": 7, "time": 7, "breakdown": 7, "p": [8, 11, 13], "pacificu": 8, "pharynx": 8, "refer": [9, 11], "simul": [10, 11, 13], "model": 10, "graph": [11, 12, 13], "benjamin": [11, 12, 13], "d": [11, 12, 13], "pedigo": [11, 12, 13], "These": [11, 13], "slide": [11, 13], "http": 11, "tinyurl": 11, "com": 11, "cambridg": 11, "bilater": [11, 13], "homolog": [11, 13], "why": [11, 13], "care": [11, 13], "about": [11, 13], "how": [11, 13], "can": [11, 13], "we": [11, 13], "structur": 11, "predict": [11, 13], "an": 11, "align": 11, "from": [11, 13], "anoth": 11, "when": 11, "label": 11, "aren": 11, "t": [11, 13], "help": 11, "do": [11, 13], "repres": 11, "ani": 11, "permut": 11, "equal": 11, "valid": 11, "3": [11, 13], "A": [11, 13], "respect": 11, "what": [11, 13], "frame": 11, "problem": 11, "mathemat": 11, "min_": [11, 13], "mathcal": [11, 13], "underbrac": 11, "overbrac": 11, "pbp": 11, "text": 11, "reorder": 11, "b": [11, 13], "_f": [11, 13], "2": [11, 13], "_": 11, "distanc": 11, "between": 11, "adj": 11, "mat": 11, "where": 11, "set": 11, "hard": 11, "let": 11, "s": 11, "try": 11, "out": 11, "graspolog": 11, "scipi": 11, "larval": [11, 13], "drosophila": [11, 13], "onli": 11, "perform": 11, "fairli": 11, "well": 11, "qualiti": 11, "proofread": 11, "mani": 11, "wai": 11, "improv": 11, "thi": [11, 13], "partial": 11, "known": 11, "p_": 11, "your": 11, "similar": 11, "inform": 11, "e": 11, "g": 11, "morpholog": 11, "trace": 11, "sp": 11, "edg": [11, 13], "type": 11, "multilay": 11, "1": [11, 13], "develop": 11, "thu": [11, 13], "far": [11, 13], "ve": [11, 13], "contralater": [11, 13], "ar": [11, 13], "bisect": [11, 12, 13], "dataset": [11, 13], "modern": 11, "put": 11, "togeth": 11, "combin": 11, "extra": 11, "even": 11, "more": [11, 13], "practic": 11, "consider": 11, "new": 11, "code": 11, "r": 11, "info": [11, 13], "further": 11, "read": 11, "ackowledg": 11, "question": 11, "john": 12, "hopkin": 12, "univers": 12, "increas": 12, "extens": 12, "multiplex": 12, "summari": [12, 13], "acknowledg": 12, "aka": 13, "know": 13, "re": 13, "good": 13, "doe": 13}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9, "sphinx": 56}})