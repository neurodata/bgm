import seaborn as sns

tab10 = sns.color_palette("tab10")
method_palette = dict(zip(["GM", "BGM"], [tab10[0], tab10[3]]))

set2 = sns.color_palette("Set2")
subgraph_palette = dict(zip(["LL", "RR", "LR", "RL"], set2))
