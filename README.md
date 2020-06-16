## Outcome Correlation in Graph Neural Network Regression

### This repository hosts the code and some example data for the following paper:  
[Outcome Correlation in Graph Neural Network Regression](https://arxiv.org/abs/2002.08274)  
[Junteng Jia](https://000justin000.github.io/), and [Austin R. Benson](https://www.cs.cornell.edu/~arb/)  
KDD, 2020.

There is a also a (G)pytorch [implementation](https://github.com/JunwenBai/correlation-gnn) from Junwen Bai and Yucheng Lu.

Our paper identifies the fact that GNN regression residuals are oftentimes correlated among neighboring vertices, and we propose simple and efficient algorithms to explore the correlation structure:
- C-GNN models the correlation as a multivariate Gaussian and learns the correlation structure in O(m) per optimization step, where m is the number of edges.
- LP-GNN assumes positive correlation among neighboring vertices, and runs label propagation to interpolate GNN residuals on the testing vertices.

Our code is tested under in Julia 1.0.5, you can install all dependent packages by running.
```
julia env.jl
```

### Usage
In order to use our code for your own graph regression tasks, the following four type of information is required 1) the topology of your graph as a LightGraph object 2) the adjacency matrices **A** of your graph, each matrix represent one type of edge 3) outcomes for observed vertices 4) features for all vertices.

LP-GNN algorithm requires minimal implementation overhead on top of standard GNN. The following is code snippet from [example_lpgnn.jl](examples/example_lpgnn.jl) that predicts county-level election outcomes with demographical features.
```julia
#---------------------------------------------------------------------------------------------
# read the four requirements as listed above
#---------------------------------------------------------------------------------------------
G, A, labels, feats = read_network(network_trans);
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# define and train GNN
#---------------------------------------------------------------------------------------------
# encoder that embed vertices to vector representations
enc = graph_encoder(length(feats[1]), dim_out, dim_h, repeat(["SAGE_Mean"], 2); σ=relu);
# regression layer that maps representation to prediction
reg = Dense(dim_out, 1); 
# GNN prediction 
getRegression = L -> vcat(reg.(enc(G, L, u->feats[u]))...);
# training
Flux.train!(L->mse(labels[L], getRegression(L)), params(enc, reg), mini_batches, ADAM(0.001));
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# LP-GNN testing
#---------------------------------------------------------------------------------------------
# Γ: normalized Laplacian matrix
# L: training vertices
# U: testing vertices
# rL: GNN predicted residual for testing vertices
# lU: LP-GNN predicted outcomes for testing vertices
#---------------------------------------------------------------------------------------------
pL = getRegression(L)
pU = getRegression(U)

rL = labels[L] - data(pL);
lU = pU + cg(Γ[U,U], -Γ[U,L]*rL);
#---------------------------------------------------------------------------------------------
```
In the algorithm above, only the last 4 lines differ from the standard GNN.

The C-GNN algorithm is slightly more involving since it need to optimize the framework parameters to fit the observed correlation pattern. An example is given in [example_cgnn.jl](examples/example_cgnn.jl), which only introduce tens of lines additional code comparing to the standard GNN algorithm.

In order to run the examples, you can simply use:
```julia
julia examples/example_cgnn.jl
julia examples/example_lpgnn.jl
```


### Reproduce Experiments in Paper
The experiments in our paper can be reproduced by running.
```
bash run.sh
```
which would write the outputs to [/logs](/logs).

If you have any questions, please email to [jj585@cornell.edu](mailto:jj585@cornell.edu).
