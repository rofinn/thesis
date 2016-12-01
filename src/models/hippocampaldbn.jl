"""
The full hippocampal DBN model

DG (NG):
    - visible = CA
    - conditioned on ECII
    - fit = convergence
CA:
    - visible = ECII
    - optionally conditioned on previous output
    - fit = convergence

Testing:
1) Train greedily based on the conditional dependence
    ie) (ECII,CA) -> (CA3, DG)
2) Test by cued recall.

Questions:
* What about asynchronous interactions between EC, DG and CA3?
* Shouldn't it be possible to have concurrent training and testing?
* For the CA3 recurrent condition, what happens if the previous nobs
    doesn't match the current? Do we just always fit to curr I guess?
"""
type HippocampalDBN <: AbstractModel
    dg::AbstractLayer
    ca::Conditional

    function HippocampalDBN(kwargs::Dict)
        input_length = kwargs[:visible]

        ca_kwargs = Dict([(Symbol(k), v) for (k, v) in kwargs[:ca]])
        ca_kwargs[:visible] = input_length
        ca_kwargs[:cond] = 0
        ca_kwargs[:steps] = 0

        dg_constructor = MODELS[kwargs[:dg]["type"]]
        dg_kwargs = Dict([(Symbol(k), v) for (k, v) in kwargs[:dg]])
        dg_kwargs[:visible] = ca_kwargs[:hidden]
        dg_kwargs[:cond] = input_length

        dg_kwargs[:rbm] = ConditionalRBM(
            Float64,
            Bernoulli,
            Bernoulli,
            dg_kwargs[:visible],
            dg_kwargs[:hidden],
            dg_kwargs[:cond],
            sigma=0.001
        )

        new(
            dg_constructor(dg_kwargs),
            Conditional(ca_kwargs),
        )
    end
end

"""
Trains:

1. EC <-> CA layer (optionally conditioned by time)
2. CA <-> DG layer (conditioned on EC)
"""
function fit(m::HippocampalDBN, X::Mat{Float64})
    println("Training CA: $(m.ca.rbm)")
    window = m.ca.steps + 1
    traj_X = traj(X', window)'
    println("Traj: $(size(traj_X))")
    vis = traj_X[(size(traj_X, 1) - size(X,1) + 1):end, :]
    cond = traj_X[1:(size(traj_X, 1) - size(X,1)), :]
    ca_in = vcat(vis, cond)
    fit(m.ca.rbm, ca_in, m.ca.context)
    println("Sampling ($(size(vis)))...")
    ca_out = sample_hiddens(m.ca.rbm, X)

    println("Training DG: $(m.dg.rbm)")
    #println(m.dg.context)
    fit(m.dg.rbm, vcat(ca_out, vis), m.dg.context)
end

"""
Recall a pattern by:

1. Transforming through EC -> CA
2. Generating CA <-> DG (conditioned on EC)
3. Generating CA -> EC
"""
function generate(m::HippocampalDBN, X::Array{Float64, 2})
    window = m.ca.steps + 1
    traj_X = traj(X', window)'
    vis = traj_X[(size(traj_X, 1) - size(X,1) + 1):end, :]
    cond = traj_X[1:(size(traj_X, 1) - size(X,1)), :]
    Boltzmann.dynamic_biases!(m.ca.rbm, cond)
    ca_pos = sample_hiddens(m.ca.rbm, vis)

    Boltzmann.dynamic_biases!(m.dg.rbm, vis)
    dg_pos = sample_hiddens(m.dg.rbm, ca_pos)
    ca_neg = sample_visibles(m.dg.rbm, dg_pos)

    return dg_pos, sample_visibles(m.ca.rbm, ca_neg)
end

function generate(m::HippocampalDBN, X::Array{Float64, 1})
    generate(m, reshape(X, length(X), 1))
end
