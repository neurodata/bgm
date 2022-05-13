# Bisected graph matching improves automated neuron pairing between hemispheres in connectomes

## Abstract 
Graph matching algorithms attempt to find the best correspondence between the nodes of two networks. These techniques have previously been used to match individual neurons in nanoscale connectomes; in particular, to find pairings of neurons across hemispheres. However, since graph matching techniques deal specifically with two networks, they have only utilized the ipsilateral (same hemisphere) subgraphs when performing the matching. Here, we present a modification to a state-of-the-art graph matching algorithm which allows it to solve what we call the bisected graph matching problem. This modification allows us to use connections between the brain hemispheres when predicting neuron pairs. We show in simulation as well as real connectome examples that when edge correlation is present between the contralateral (between hemisphere) subgraphs, this approach improves matching accuracy. We expect that our proposed method will improve future endeavors to accurately match neurons between hemispheres in connectomes, and be useful in other applications where the bisected graph matching problem arises.

## Creating an environment (for me at least)
`poetry env use python3.9`
`source ./.venv/bin/activate`
`poetry install`
