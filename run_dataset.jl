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

include("read_network.jl");
include("utils.jl");

function run_dataset(network_trans, network_ind, accuracyFun, regressor="zero", correlation="zero", inductive=false)
    @assert regressor in ["zero", "linear", "mlp", "gnn"]
    @assert correlation in ["zero", "homo", "learned"]

    dim_out, dim_h = 8, 16;
    t, k, num_steps = 128, 32, 1500;
    num_ave = 10;

    ptr = 0.60;
    rrt = Vector();
     αt = Vector();
     βt = Vector();

    if inductive
        ptr_inductive = 0.00:0.05:0.60;
        rri = [[Vector() for _ in 1:length(ptr_inductive)] for _ in 1:length(network_ind)];
    end

    function run_once(seed_val)
        Random.seed!(seed_val);
        println("\n\nseed_val:    ", seed_val);

        G, A, labels, feats = read_network(network_trans); n = nv(G);

        d = sum(sum(A), dims=1)[:];
        S = [spdiagm(0=>d.^-0.5)*A_*spdiagm(0=>d.^-0.5) for A_ in A];

        L, VU = rand_split(n, ptr);
        V, U = VU[1:div(length(VU),2)], VU[div(length(VU),2)+1:end];

        ab = param(vcat(zeros(length(A)), 3.0));
        getα() = tanh.(ab[1:end-1]);
        getβ() = exp(ab[end]);

        if regressor == "zero"
            getRegression = L -> zeros(length(L));
            θ = params();
            optθ = ADAM(0.0);
        elseif regressor == "linear"
            lls = Dense(length(feats[1]), 1);
            getRegression = L -> vcat(lls.([feats[u] for u in L])...);
            θ = params(lls);
            optθ = ADAM(0.1);
        elseif regressor == "mlp"
            mlp = Chain(Dense(length(feats[1]), dim_h, relu), Dense(dim_h, dim_h, relu), Dense(dim_h, dim_out, relu), Dense(dim_out, 1));
            getRegression = L -> vcat(mlp.([feats[u] for u in L])...);
            θ = params(mlp);
            optθ = ADAM(0.001);
        elseif regressor == "gnn"
            enc = graph_encoder(length(feats[1]), dim_out, dim_h, repeat(["SAGE_Mean"], 2); σ=relu);
            reg = Dense(dim_out, 1);
            getRegression = L -> vcat(reg.(enc(G, L, u->feats[u]))...);
            θ = params(enc, reg);
            optθ = ADAM(0.001);
        else
            error("unexpected regressor type");
        end

        if match(r"twitch", network_trans) != nothing
            optφ = "l_bfgs";
            φ_skip = 100;
        else
            optφ = Descent(0.1);
            φ_skip = 10;
        end

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

        function call_back()
            @printf("%6.3f,    %6.3f,    [%s],    %6.3f\n",
                    accuracyFun(labels[L], dat(pred(L,V; G=G,labels=labels,predict=getRegression,α=((correlation == "homo") ? ones(length(A)) : getα()),β=getβ(),S=S))),
                    accuracyFun(labels[V], dat(pred(V,L; G=G,labels=labels,predict=getRegression,α=((correlation == "homo") ? ones(length(A)) : getα()),β=getβ(),S=S))),
                    array2str(getα()),
                    getβ());
        end

        mini_batch_size = Int(round(length(L) * 0.05));
        mini_batches = [sample(L, mini_batch_size, replace=false) for _ in 1:num_steps];
        train!(loss, getrL, getΩ,
               (correlation == "learned") ? true : false,
               θ,
               params(ab),
               mini_batches, L,
               optθ, optφ;
               cb=call_back, φ_start=0, φ_skip=φ_skip, cb_skip=100);

        push!(rrt, accuracyFun(labels[U], dat(pred(U,L; G=G,labels=labels,predict=getRegression,α=((correlation == "homo") ? ones(length(A)) : getα()),β=getβ(),S=S))));
        push!(αt, getα());
        push!(βt, getβ());
        @printf("\n pctg,    rr,    α,    β\n");
        @printf("%6.3f,    %6.3f,    [%s],    %6.3f\n", ptr, rrt[end], array2str(getα()), getβ());

        if inductive
            if regressor == "gnn" && correlation == "zero"
                @save "logs/gnn.bson" enc reg ab;
            elseif regressor == "mlp" && correlation == "zero"
                @save "logs/mlp.bson" mlp ab;
            end

            for i in 1:length(network_ind)
                Random.seed!(seed_val);
                G, A, labels, feats = read_network(network_ind[i]); n = nv(G);
                d = sum(sum(A), dims=1)[:];
                S = [spdiagm(0=>d.^-0.5)*A_*spdiagm(0=>d.^-0.5) for A_ in A];

                @printf("\n pctg_inductive,    rr,    α,    β\n");
                for j in 1:length(ptr_inductive)
                    L, VU = rand_split(n, ptr_inductive[j]);
                    V, U = VU[1:div(length(VU),2)], VU[div(length(VU),2)+1:end];

                    if regressor in ["mlp", "gnn"] && correlation == "zero" && length(L) > 0
                        mini_batch_size = Int(round(length(L) * 0.05));
                        mini_batches = [sample(L, mini_batch_size, replace=false) for _ in 1:500];

                        if regressor == "mlp"
                            @load "logs/mlp.bson" mlp ab;
                            θ = params(mlp);
                        elseif regressor == "gnn"
                            @load "logs/gnn.bson" enc reg ab;
                            θ = params(enc, reg);
                        end

                        train!(loss, getrL, getΩ,
                               false,
                               θ,
                               params(ab),
                               mini_batches, L,
                               ADAM(0.0005), Descent(0.1);
                               cb=call_back, φ_start=0, φ_skip=φ_skip, cb_skip=100);
                    end

                    push!(rri[i][j], accuracyFun(labels[VU], dat(pred(VU,L; G=G,labels=labels,predict=getRegression,α=((correlation == "homo") ? ones(length(A)) : getα()),β=getβ(),S=S))));
                    @printf("%6.3f,    %6.3f,    [%s],    %6.3f\n", ptr_inductive[j], rri[i][j][end], array2str(getα()), getβ());
                end
            end
        end
    end

    for seed_val in 1:num_ave
        run_once(seed_val);
    end

    @printf("\n trans: %s,    %6.3f ± %6.3f\n", network_trans, mean(rrt), std(rrt));
    @printf(" α, β: [%s] ± [%s], %6.3f ± %6.3f\n", array2str(mean(αt)), array2str(std(αt)), mean(βt), std(βt));
    if inductive
        for i in 1:length(network_ind)
            @printf(" ind: %s,    [%s] ± [%s]\n", network_ind[i], array2str(mean.(rri[i])), array2str(std.(rri[i])));
        end
    end
end
