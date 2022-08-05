import seaborn as sns

# tab10 = sns.color_palette("tab10")
# tab10_pastel = sns.color_palette("pastel")
# method_palette = dict(zip(["GM", "BGM"], [tab10_pastel[0], tab10[3]]))
# tab10 = sns.color_palette("tab10")

tab10_colorblind = sns.color_palette("colorblind")
method_palette = dict(zip(["GM", "BGM"], [tab10_colorblind[0], tab10_colorblind[1]]))

set2 = sns.color_palette("Set2")
subgraph_palette = dict(zip(["LL", "RR", "LR", "RL"], set2))

dashes = {"GM": (3, 1), "BGM": ""}
