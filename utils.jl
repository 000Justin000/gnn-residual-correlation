using Random;
using StatsBase;
using LightGraphs;
using LinearAlgebra
using SparseArrays;
using Arpack;
using Flux;
using Flux.Tracker: data, track, @grad, forward, Params, update!, back!, grad, gradient;
using Flux.Tracker: TrackedReal, TrackedVector, TrackedMatrix;
using MLBase: roc, f1score;
using Printf;
using PyCall;

function mBCG(mmm_A::Function, B::Array{Float64,2}; PC::Function=Y->Y, k::Int=size(B,1), tol=1.0e-6)
    """
    Args:
     mmm_A: matrix matrix multiplication routine
         B: right-hand-side vectors
        PC: apply preconditioner to each column of the matrix
         k: max # of iterations
       tol: error tolerance

    Returns:
         X: solution vectors
        TT: Lanczos tridiagonal matrices
    """

    n,t = size(B);
    X = zeros(n,t);
    R = B - mmm_A(X);
    Z = PC(R);
    P = Z;
    α = zeros(t);
    β = zeros(t);

    T = [(dv=Vector{Float64}(), ev=Vector{Float64}()) for _ in 1:t];

    tol_vec = tol .+ tol*sqrt.(sum(R.*R, dims=1)[:]);
    for j in 1:k
        if all(sqrt.(sum(R.*R, dims=1)[:]) .< tol_vec)
            break;
        end

        AP = mmm_A(P);
        α_ = sum(R.*Z, dims=1)[:] ./ sum(P.*AP, dims=1)[:];
        X_ = X + P .* α_';
        R_ = R - AP .* α_';
        Z_ = PC(R_);
        β_ = sum(Z_.*R_, dims=1)[:] ./ sum(Z.*R, dims=1)[:];
        P_ = Z_ + P .* β_';

        for i in 1:t
            if j == 1
                push!(T[i].dv, 1.0/α_[i]);
            else
                push!(T[i].dv, 1.0/α_[i]+β[i]/α[i]);
                push!(T[i].ev, sqrt(β[i])/α[i]);
            end
        end

        P = P_;
        R = R_;
        Z = Z_;
        X = X_;
        α = α_;
        β = β_;
    end

    return X, [SymTridiagonal(dv,ev) for (dv,ev) in T];
end

function R2(y, y_)
    """
    Args:
         y: true labels
        y_: predicted labels

    Return:
        coefficients of determination
    """

    return 1.0 - sum((y_ .- y).^2.0) / sum((y .- mean(y)).^2.0);
end

function sign_accuracy(y, y_)
    """
    Args:
         y: true labels
        y_: predicted labels

    Return:
        accuracy in binary classification setting
    """

    return sum(sign.(y_.+eps()) .== sign.(y.+eps())) / length(y);
end

function F1(y, y_)
    """
    Args:
         y: true labels
        y_: predicted labels

    Return:
        harmonic mean of precision and recall
    """

    return f1score(roc(convert(Vector{Int}, sign.(data(y))), convert(Vector{Int}, sign.(data(y_)))));
end

function array2str(arr)
    """
    Args:
       arr: array of data
       fmt: format string
    Return:
       string representation of the array
    """

    (typeof(arr[1]) <: String) || (arr = map(x->@sprintf("%10.3f", x), arr));
    return join(arr, ", ");
end

function getΓ(α, β; S)
    return β * I - β * sum([α_*S_ for (α_,S_) in zip(α, S)]);
end

function get∂Γ∂α(α, β; S)
    return -[β*S_ for S_ in S];
end

function get∂Γ∂β(α, β; S)
    return I - sum([α_*S_ for (α_,S_) in zip(α, S)]);
end

logdetΓ(α::TrackedVector, β::TrackedReal; S, P, t, k) = track(logdetΓ, α, β; S=S, P=P, t=t, k=k);
@grad function logdetΓ(α, β; S, P, t, k)
    """
    Args:
         α: model parameter vector
         β: model parameter
         S: normalized adjacency matrix vector
         P: index set
         t: # of trial vectors
         k: # of Lanczos tridiagonal iterations

    Return:
         log determinant of the principle submatrix
    """

    α = data(α);
    β = data(β);

    n = length(P);
    Z = randn(n,t);

    Γ = getΓ(α, β; S=S);
    ∂Γ∂α = get∂Γ∂α(α, β; S=S);
    ∂Γ∂β = get∂Γ∂β(α, β; S=S);

    X, TT = mBCG(Y->Γ[P,P]*Y, Z; k=k);

    vv = 0;
    for T in TT
        eigvals, eigvecs = eigen(T);
        vv += sum(eigvecs[1,:].^2 .* log.(eigvals));
    end

    Ω = vv*n/t;

    trΓiM(M) = sum(X.*(M[P,P]*Z))/t;
    ∂Ω∂α = map(trΓiM, ∂Γ∂α);
    ∂Ω∂β = trΓiM(∂Γ∂β);

    return Ω, Δ -> (Δ*∂Ω∂α, Δ*∂Ω∂β)
end

function test_logdetΓ(n=100)
    G = random_regular_graph(n, 3);
    A = adjacency_matrix(G);
    D = spdiagm(0=>sum(A,dims=1)[:]);
    S = [D^-0.5 * A * D^-0.5];

    #------------------------
    # true value
    #------------------------
    α = param(tanh.([1.0]));
    β = param(softplus(1.0));
    #------------------------
    Γ = getΓ(α, β; S=S);
    Ω = logdet(Matrix(Γ));
    #------------------------
    Tracker.back!(Ω, 1);
    @printf("accurate:       [%s],    %6.3f\n", array2str(Tracker.grad(α)), Tracker.grad(β));
    #------------------------

    #------------------------
    # approximation
    #------------------------
    α = param(tanh.([1.0]));
    β = param(softplus(1.0));
    #------------------------
    Ω = logdetΓ(α, β; S=S, P=collect(1:n), t=50, k=n);
    #------------------------
    Tracker.back!(Ω, 1);
    @printf("approximate:    [%s],    %6.3f\n", array2str(Tracker.grad(α)), Tracker.grad(β));
    #------------------------
end

quadformSC(α::TrackedVector, β::TrackedReal, rL; S, L) = track(quadformSC, α, β, rL; S=S, L=L);
@grad function quadformSC(α, β, rL; S, L)
    """
    Args:
         α: model parameter vector
         β: model parameter
        rL: noise on vertex set L
         S: normalized adjacency matrix vector
         L: index set

    Return:
         quadratic form: rL' (ΓLL - ΓLU ΓUU^-1 ΓUL) rL
    """

    α = data(α);
    β = data(β);
    rL = data(rL);

    Γ = getΓ(α, β; S=S);
    ∂Γ∂α = get∂Γ∂α(α, β; S=S);
    ∂Γ∂β = get∂Γ∂β(α, β; S=S);

    U = setdiff(1:size(S[1],1), L);

    Ω = rL'*Γ[L,L]*rL - rL'*Γ[L,U]*cg(Γ[U,U],Γ[U,L]*rL);

    quadform_partials(M) = rL'*M[L,L]*rL - rL'*M[L,U]*cg(Γ[U,U],Γ[U,L]*rL) + rL'*Γ[L,U]*cg(Γ[U,U],M[U,U]*cg(Γ[U,U],Γ[U,L]*rL)) - rL'*Γ[L,U]*cg(Γ[U,U],M[U,L]*rL);
    ∂Ω∂α = map(quadform_partials, ∂Γ∂α);
    ∂Ω∂β = quadform_partials(∂Γ∂β);
    ∂Ω∂rL = 2*Γ[L,L]*rL - 2*Γ[L,U]*cg(Γ[U,U],Γ[U,L]*rL);

    return Ω, Δ -> (Δ*∂Ω∂α, Δ*∂Ω∂β, Δ*∂Ω∂rL);
end

function test_quadformSC(n=100)
    G = random_regular_graph(n, 3);
    A = adjacency_matrix(G);
    D = spdiagm(0=>sum(A,dims=1)[:]);
    S = [D^-0.5 * A * D^-0.5];
    L = randperm(n)[1:div(n,2)];
    U = setdiff(1:n, L);
    rL_ = randn(div(n,2))

    #------------------------
    # true value
    #------------------------
    α = param(tanh.([1.0]));
    β = param(softplus(1.0));
    rL = param.(rL_);
    #------------------------
    Γ = getΓ(α, β; S=S);
    SC = Γ[L,L] - Γ[L,U]*inv(Matrix(Γ[U,U]))*Γ[U,L];
    Ω = rL' * SC * rL;
    #------------------------
    Tracker.back!(Ω, 1);
    @printf("accurate:       [%s],    %6.3f,    [%s]\n", array2str(Tracker.grad(α)), Tracker.grad(β), array2str(Tracker.grad.(rL)[1:10]));
    #------------------------

    #------------------------
    # approximation
    #------------------------
    α = param(tanh.([1.0]));
    β = param(softplus(1.0));
    rL = param(rL_);
    #------------------------
    Ω = quadformSC(α, β, rL; S=S, L=L);
    #------------------------
    Tracker.back!(Ω, 1);
    @printf("approximate:    [%s],    %6.3f,    [%s]\n", array2str(Tracker.grad(α)), Tracker.grad(β), array2str(Tracker.grad(rL)[1:10]));
    #------------------------
end

function rand_split(n, ptr)
    """
    Args:
         n: total number of data points
       ptr: percentage of training data

    Returns:
         L: indices for training data points
         U: indices for testing data points
    """

    randid = randperm(n);
    ll = Int64(ceil(ptr*n));

    L = randid[1:ll];
    U = randid[ll+1:end];

    return L, U;
end

function expansion(m, ids)
    """
    Args:
         m: overall dimension
       ids: a length m_ vector with indices indicating location

    Returns:
         Ψ: a m x m_ matrix that expand a vector of dimension m_ to a vector of dimension m
    """

    m_ = length(ids);

    II = Vector{Int}();
    JJ = Vector{Int}();
    VV = Vector{Float64}();

    for (i,id) in enumerate(ids)
        push!(II, id);
        push!(JJ, i);
        push!(VV, 1.0);
    end

    return sparse(II, JJ, VV, m, m_);
end

function interpolate(L, rL; Γ)
    """
    Args:
         L: mini_batch indices for estimating noise
        rL: noise over the mini_batch L
         Γ: precision matrix for Gaussian noise

    Returns:
         r: noise over all vertices
    """

    n = size(Γ,1);
    U = setdiff(1:n, L);
    rU = cg(Γ[U,U], -Γ[U,L]*rL);

    r = expansion(n,L) * rL + expansion(n,U) * rU;

    return r;
end

function pred(U, L; G, labels, predict, α, β, S)
    """
    Args:
          U: mini_batch indices for training
          L: mini_batch indices for estimating noise
     labels: vertex labels
    predict: base predictor function
          α: model parameter vector
          β: model parameter
          S: normalized adjacency matrix vector

    Returns:
         lU: predictive label = base predictor output + estimated noise
    """

    Γ = getΓ(data(α), data(β); S=S);
    pUL = predict(vcat(U, L));
    pU = pUL[1:length(U)];
    pL = pUL[length(U)+1:end];

    rL = labels[L] - data(pL);
    lU = pU + interpolate(L, rL; Γ=Γ)[U];

    return lU;
end

function train!(loss, getrL, getΩ, updateφ, θ, φ, mini_batches, L, optθ, optφ; cb=()->(), φ_start=1, φ_skip=1, cb_skip=1)
    """
    extend training method to allow using different optimizers for different parameters
    """

    # this is a dummy optimizer to destroy gradients
    opt0 = Descent(0.0);
    θφ = Params(vcat(collect(θ), collect(φ)));
    for (i,mini_batch) in enumerate(mini_batches)
        gsθφ = gradient(() -> loss(mini_batch), θφ);
        update!(optθ, θ, gsθφ);
        update!(opt0, φ, gsθφ);

        if updateφ && (i >= φ_start) && (i % φ_skip == 0)

            # l-bfgs optimizer
            if optφ == "l_bfgs"
                rL = data(getrL(L));
                function obj_func(p)
                    ab = param(p);
                    Ω = getΩ(tanh.(ab[1:end-1]), exp(ab[end]), rL, L, true);
                    back!(Ω);

                    return data(Ω), grad(ab);
                end

                so = pyimport("scipy.optimize");
                φx = data(collect(φ)[1]);
                Ωx = obj_func(φx)[1];
                φdesign = [φx, vcat(-3.0*ones(length(φx)-1), 3.0), vcat(3.0*ones(length(φx)-1), 3.0)]

                for φ0 in φdesign
                    opt = so.fmin_l_bfgs_b(obj_func, φ0, bounds=repeat([(-7.0, 7.0)], length(φ0)));
                    if opt[2] < Ωx
                        φx, Ωx = opt[1], opt[2];
                    end
                end

                n_trials = ((i - φ_skip) >= φ_start ? 3 : 3);
                for _ in 1:n_trials
                    opt = so.fmin_l_bfgs_b(obj_func, randn(length(φx)), bounds=repeat([(-7.0, 7.0)], length(φx)));
                    if opt[2] < Ωx
                        φx, Ωx = opt[1], opt[2];
                    end
                end

                update!(collect(φ)[1], φx .- data(collect(φ)[1]));
            else
                # gradient descent optimizer
                gsθφ = gradient(() -> loss(L; logdet=true), θφ);
                update!(opt0, θ, gsθφ);
                update!(optφ, φ, gsθφ);
            end
        end

        if i % cb_skip == 0
            cb();
        end
    end
end
