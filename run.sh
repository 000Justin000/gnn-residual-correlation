# julia run.jl zero   homo    false 2>&1 | tee logs/trans_lp    &
# julia run.jl mlp    zero    false 2>&1 | tee logs/trans_mlp   &
# julia run.jl mlp    homo    false 2>&1 | tee logs/trans_lpmlp &
# julia run.jl mlp    learned false 2>&1 | tee logs/trans_cmlp  &
julia run.jl gnn    zero    false 2>&1 | tee logs/trans_gnn   &
julia run.jl gnn    homo    false 2>&1 | tee logs/trans_lpgnn &
julia run.jl gnn    learned false 2>&1 | tee logs/trans_cgnn  &
# julia run.jl mlp    zero    true  2>&1 | tee logs/ind_mlp     &
# julia run.jl gnn    zero    true  2>&1 | tee logs/ind_gnn     &
# julia run.jl gnn    learned true  2>&1 | tee logs/ind_cgnn    &
