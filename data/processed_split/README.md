# Processed data
This folder contains datasets processed using `scripts/process_c_elegans.py`, `scripts/process_maggot.py`, and `scripts/process_p_pacificus.py`.

Each connectome has a `_nodes.csv` file with information about each neuron and its 
pair designation, and an `_edgelist.csv` stored in `{source}, {target}, {weight}` format.
The weight is the number of chemical synapses from source to target.