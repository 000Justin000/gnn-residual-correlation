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

cb_skip = 100;

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

# framework parameters α, β is function of ab
ab = param(vcat(zeros(length(A)), 3.0));
getα() = tanh.(ab[1:end-1]);
getβ() = exp(ab[end]);

# encoder that embed vertices to vector representations
enc = graph_encoder(length(feats[1]), dim_out, dim_h, repeat(["SAGE_Mean"], 2); σ=relu);
# regression layer that maps representation to prediction
reg = Dense(dim_out, 1);

# GNN prediction 
getRegression = L -> vcat(reg.(enc(G, L, u->feats[u]))...);

# GNN weights, optimizer
θ = params(enc, reg);
optθ = ADAM(0.001);

# compute residual
getrL(L) = labels[L] - getRegression(L);

function getΩ(α, β, rL, L, logdet)
    Ω = quadformSC(α, β, rL; S=S, L=L);
    logdet && (Ω -= (logdetΓ(α, β; S=S, P=collect(1:nv(G)), t=t, k=k) - logdetΓ(α, β; S=S, P=setdiff(1:nv(G),L), t=t, k=k)));
    return Ω;
end

function loss(L; getα=getα, getβ=getβ, logdet=false)
    rL = getrL(L);
    Ω = getΩ(getα(), getβ(), rL, L, logdet);
    return Ω / length(L);
end

dat(x) = data.(data(x));

n_step = 0;
function call_back()
    global n_step += cb_skip;
    @printf("%6d,    %6.3f,    %6.3f,    [%s],    %6.3f\n",
            n_step,
            R2(labels[L], dat(pred(L,V; G=G,labels=labels,predict=getRegression,α=getα(),β=getβ(),S=S))),
            R2(labels[V], dat(pred(V,L; G=G,labels=labels,predict=getRegression,α=getα(),β=getβ(),S=S))),
            array2str(getα()),
            getβ());
end

# subsampled mini-batches
mini_batch_size = Int(round(length(L) * 0.05));
mini_batches = [sample(L, mini_batch_size, replace=false) for _ in 1:num_steps];

@printf("\n #steps,    training R2,    validation R2,    α,    β\n");
train!(loss, getrL, getΩ,
       true,
       θ,
       params(ab),
       mini_batches, L,
       optθ, 
       Descent(0.1);
       cb=call_back, φ_start=0, φ_skip=10, cb_skip=cb_skip);

@printf("\n transductive: 2012 -> 2012");
@printf("\n C-GNN testing R2:    %6.3f\n", R2(labels[U], dat(pred(U,L; G=G,labels=labels,predict=getRegression,α=getα(),β=getβ(),S=S))));

# inductive learning pipeline
_, _, labels, feats = read_network(network_ind);

@printf("\n inductive: 2012 -> 2016");
@printf("\n C-GNN testing R2:    %6.3f\n", R2(labels[U], dat(pred(U,L; G=G,labels=labels,predict=getRegression,α=getα(),β=getβ(),S=S))));
