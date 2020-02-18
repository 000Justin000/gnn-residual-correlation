using Random;
using Statistics;
using StatsBase: sample, randperm, mean;
using LinearAlgebra;
using SparseArrays;
using IterativeSolvers;
using LightGraphs;
using Flux;
using GraphSAGE;
using BSON: @save, @load;
using Printf;

include("../read_network.jl");
include("../utils.jl");

# train on 2012 election
# transductive accuracy: test on remaining 2012 data, test on 2012 election data
# inductive accuracy:    test on 2016 data
network_trans = "county_election_2012";
network_ind   = "county_election_2016";

# dim_out: dimension for vector representation for each vertex
# dim_h: hidden layer dimension used by GNN
# t: number of trials in stochastic estimation
# k: number of cg iteration
# num_steps: number of training steps
# ptr: fraction of training data
dim_out, dim_h = 8, 16;
t, k, num_steps = 128, 32, 1500;
ptr = 0.6;

# G: graph, light graph
# A: adjacency matrix, decomposed by edge type
# labels: margin of vectory
# feats: features on each vertex
# d: degree of each vertex
# S: normalized adjacency matrix, decomposed by edge type
G, A, labels, feats = read_network(network_trans);
n = nv(G);
d = sum(sum(A), dims=1)[:];
S = [spdiagm(0=>d.^-0.5)*A_*spdiagm(0=>d.^-0.5) for A_ in A];

# L: training vertices
# V: validation vertices
# L: testing vertices
L, VU = rand_split(n, ptr);
V, U = VU[1:div(length(VU),2)], VU[div(length(VU),2)+1:end];

# encoder that embed vertices to vector representations
enc = graph_encoder(length(feats[1]), dim_out, dim_h, repeat(["SAGE_Mean"], 2); σ=relu);
# regression layer that maps representation to prediction
reg = Dense(dim_out, 1);

# GNN prediction 
getRegression = L -> vcat(reg.(enc(G, L, u->feats[u]))...);

# GNN weights, optimizer
θ = params(enc, reg);
optθ = ADAM(0.001);

function loss(L)
    return Flux.mse(labels[L], getRegression(L));
end

dat(x) = data.(data(x));

n_step = 0;
function call_back()
    global n_step += 1;
    mod(n_step, 100) == 0 && @printf("%6d,    %6.3f,    %6.3f\n",
                                     n_step,
                                     R2(labels[L], dat(getRegression(L))),
                                     R2(labels[V], dat(getRegression(V))));
end

# subsampled mini-batches
mini_batch_size = Int(round(length(L) * 0.05));
mini_batches = [tuple(sample(L, mini_batch_size, replace=false)) for _ in 1:num_steps];

@printf("\n #steps,    training R2,    validation R2\n");
Flux.train!(loss, θ, mini_batches, optθ; cb=call_back);

@printf("\n transductive: 2012 -> 2012");
@printf("\n    GNN testing R2:    %6.3f",   R2(labels[U], dat(getRegression(U))));
@printf("\n LP-GNN testing R2:    %6.3f\n", R2(labels[U], dat(pred(U,L; G=G,labels=labels,predict=getRegression,α=ones(length(A)),β=1.0,S=S))));

# inductive learning pipeline
_, _, labels, feats = read_network(network_ind);

@printf("\n inductive: 2012 -> 2016");
@printf("\n    GNN testing R2:    %6.3f",   R2(labels[U], dat(getRegression(U))));
@printf("\n LP-GNN testing R2:    %6.3f\n", R2(labels[U], dat(pred(U,L; G=G,labels=labels,predict=getRegression,α=ones(length(A)),β=1.0,S=S))));
