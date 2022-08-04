---
marp: true
theme: slides
size: 16:9
paginate: true
---
<!-- _paginate: false -->

![bg center blur:3px opacity:15%](./../../images/background.svg)




<!-- # Generative network modeling reveals a first quantitative definition of bilateral symmetry exhibited by a whole insect brain connectome -->
<!-- ![icon](../../images/twitter.png) [_@bpedigod (Twitter)_](https://twitter.com/bpedigod) -->


<!-- # Generative network modeling reveals a quantitative definition of bilateral symmetry exhibited by a whole insect brain connectome -->
# Graph matching for connectomics

<br>

<div class="columns">
<div>

## Benjamin D. Pedigo
(he/him)
[NeuroData lab](https://neurodata.io/)
Johns Hopkins University
Dept. of Biomedical Engineering


![icon](../../images/email.png) [bpedigo@jhu.edu](mailto:bpedigo@jhu.edu)
![icon](../../images/github.png) [@bdpedigo (Github)](https://github.com/bdpedigo)
![icon](../../images/twitter.png) [@bpedigod (Twitter)](https://twitter.com/bpedigod)
![icon](../../images/web.png) [bdpedigo.github.io](https://bdpedigo.github.io/)

</div>
<div>

### These slides at: 
<!-- ### [tinyurl.com/princeton-bilarva](https://tinyurl.com/princeton-bilarva) -->


</div>
</div>

---

<!-- Timing: 23:00  -->

# Bilaterally homologous neuron pairs 

We believe a matching exists!

![center](./../../images/mbon-expression.jpg)

<!-- _footer: Eschbach et al. eLife (2021) -->


---

# Larval _Drosophila_ (AKA a maggot) brain connectome


<div class="columns">
<div>

<!-- See [Michael Windings's talk](https://conference.neuromatch.io/abstract?edition=2021-4&submission_id=recVeh4RZFFRAQnIo) -->
- Collaboration with Marta Zlatic/Albert Cardona's groups - led by Michael Winding
- First whole-brain, single-cell connectome of any insect
- ~3000 neurons, ~550K synapses
- Both hemispheres of the brain reconstructed

</div>
<div>

![](./../../images/Figure1-brain-render.png)

</div>
</div>

<footer>
Winding, Pedigo et al. “The complete connectome of an insect brain.” In prep. (2022)
</footer>

---

# Neuron pairs 

<div class="columns2-bl">
<div>

![center h:550](./../../images/match-neuron-morphology.png)

</div>
<div>

![center h:350](./../../images/pair-rank-panel.png)

</div>
</div>

---
# Why care about the pairs? 
- Understand stereotypy
- Proofreading
- Downstream analysis 
   - Omnibus embedding, for example
- Useful to collapse complexity by ~2 

---

# Can we use networks to predict this matching?

![center h:500](../../images/the-wire.png)

<!-- _footer: The Wire, Season 3, Episode 7 -->

--- 
# Graph matching - what is it?

![center](../../images/network-matching-explanation.svg)

---
# Graph matching - how do we know if we're doing good?

<style scoped>
h2 {
    justify-content: center;
    text-align: center;
}
</style>

## $\min_{P \in \mathcal{P}} \|A - P B P^T\|_F^2$

<div class="columns2-bl">
<div>

- $\mathcal{P}$ is the set of permutation matrices
- Measures the number of edge disagreements for unweighted 
- Norm of edge disagreements for weighted


</div>
<div>

![h:300](./../../images/perm-matrix.png)

</div>
</div>

---

# Graph matching - how do we do it?

<style scoped>
h2 {
    justify-content: center;
    text-align: center;
}
</style>

## $\min_{P \in \mathcal{P}} \|A - P B P^T\|_F^2$
- Search space is not convex:
    - $0.5 P_1 + 0.5 P_2 \notin \mathcal{P}$
- Relax to the "birkoff polytope:" doubly stochastic matrices
    - Rows and columns all sum to 1, but not all elements have to be 1 
    - Similar to *transport* as opposed to *assignment* problems: we are still mapping nodes between networks, but this is a *soft* mapping
- Apply Frank-Wolfe method
    - Minimize a first-order Taylor series of the objective function over a convex set
    - Requires a gradient $\nabla f(P)$

<!-- _footer: Vogelstein et al. 2015 - Fast approximate quadratic (FAQ) algorithm -->

---
# How does it do on the maggot brain? 

<div class="columns2-bl">
<div>

![center h:400](./../../_build/html/_images/left-pair-predictions.svg)

</div>
<div>

- ~76-80% correct on the full brain connectome for left-right pair prediction
- Lots of bells and whistles one can add:
    - Edge types
    - Seeds
    - Morphology

</div>
</div>

--- 

<style scoped>
h1, h2 {
    padding-top: 140px;
    justify-content: center;
    text-align: center;
}
</style>

# Thus far, we've not used the contralateral connections

## These are about 1/3 of the edges in the brain!

---
# From graph matching to bisected graph matching

![](../../../results/figs/explain/explain.svg)

<!-- --LL: #66c2a5;
    --RR: #fc8d62;
    --source: #8da0cb;
    --target: #e78ac3;
    --LR: #8da0cb;
    --RL: #e78ac3; -->

New gradient: 

$$\nabla f(P) = - \textcolor{#66c2a5}{A_{LL}} P \textcolor{#fc8d62}{A_{RR}}^T + \textcolor{#66c2a5}{A_{LL}}^TP\textcolor{#fc8d62}{A_{RR}} + \textcolor{#8da0cb}{A_{LR}} P^T \textcolor{#e78ac3}{A_{RL}}^T + \textcolor{#e78ac3}{A_{RL}}^T P^T \textcolor{#8da0cb}{A_{LR}}$$

---
# Simulations

<div class="columns">
<div>

- Created simulated networks with two "sides"
   - Ipsilateral subgraphs had edge correlation of 0.8
   - Contralateral subgraphs had varying correlation (x-axis at right)
- Measured matching accuracy for graph matching (GM) and bisected graph matching (BGM)

</div>
<div>

![](../../../results/figs/simulations/match_ratio_by_contra_rho.svg)

</div>
</div>

--- 
# Connectome datasets

![center h:550](./../../../results/figs/connectomes/match_accuracy_comparison.svg)

---
# Summary 

![bg center blur:3px opacity:15%](./../../images/background.svg)

- Modified the graph matching problem to allow us to incorporate information from contralateral connections in the brain
- Proposed an algorithm to solve the problem, which amounts to a simple modification of previous graph matching tools (FAQ)
- Showed in simulation that with sufficient correlation in contralateral edges, this 
  can help with matching
- Demonstrated that this indeed improves matching in five different bilateral connectome
  datasets 

---
# More info 

![bg center blur:3px opacity:15%](./../../images/background.svg)

<div class="columns">
<div>

- Tutorial on graph matching [https://bdpedigo.github.io/networks-course/graph_matching.html](https://bdpedigo.github.io/networks-course/graph_matching.html)
- Code in graspologic [https://microsoft.github.io/graspologic/latest/reference/reference/match.html](https://microsoft.github.io/graspologic/latest/reference/reference/match.html)
- Code in SciPy [https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.quadratic_assignment.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.quadratic_assignment.html)
  
Big thanks to Ali Saad-Eldin for writing most of the code!

</div>
<div>

- This repo: [https://github.com/neurodata/bgm](https://github.com/neurodata/bgm)
- This Jupyter Book: [http://docs.neurodata.io/bgm/abstract.html](http://docs.neurodata.io/bgm/abstract.html)
- This manuscript: [https://www.biorxiv.org/content/10.1101/2022.05.19.492713](https://www.biorxiv.org/content/10.1101/2022.05.19.492713)

</div>
</div>

<!-- _footer:  -->