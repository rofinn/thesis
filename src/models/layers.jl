import Boltzmann:
    RBM,
    grad_apply_momentum!,
    update_weights!,
    hid_means,
    report,
    Mat,
    ConditionalRBM

import DataFrames: DataFrame

abstract AbstractLayer <: AbstractModel

type LayerMonitor <: Monitor
    samples::Int
    scores::Array{Float64}
    hsize::Array{Int}
    runtimes::Array{Float64}


    function LayerMonitor(samples=10)
        new(samples, Float64[], Int[], Float64[])
    end
end

function report(m::LayerMonitor, rbm::AbstractRBM,
                          epoch::Int, epoch_time::Float64,
                          score::Float64)
    push!(m.scores, score)
    push!(m.runtimes, epoch_time)
    push!(m.hsize, length(rbm.hbias))
end

function update!(m::LayerMonitor, layer::AbstractLayer, X::Array)
    scorer = get(layer.context, :scorer, Boltzmann.pseudo_likelihood)
    score = Variance()
    runtime = Variance()

    for i in 1:m.samples
        tic()
        s = scorer(layer.rbm, X)
        println(s)
        fit!(score, s)
        fit!(runtime, toq())
    end

    result = (length(m.scores)+1, mean(runtime), mean(score))
    report(m, layer.rbm, result...)
    return result
end

function DataFrame(m::LayerMonitor)
    return DataFrame(
        Iteration = 1:length(m.scores),
        Scores = m.scores,
        Size = m.hsize,
        Runtime = m.runtimes
    )
end

const DEFAULT_APOP_WEIGHTING = Dict(
    :age => 0.15,
    :diff => 0.65,
    :act => 0.2
)

for i in [ "baserbm", "neurogen"]
    include(joinpath("layers", "$(i).jl"))
end

"""
Our gompertz function assumes time
values between 0 and 1.
"""
function gompertz(time::Array{Float64}, shape=5)
    t = (time * 2) - 1
    result = e .^ -e .^ -(shape * t)

    return result
end

function fit(layer::AbstractLayer, X::Array)
    Boltzmann.fit(
        layer.rbm,
        X,
        layer.context
    )
end

function generate(layer::AbstractLayer, X::Array)
    sampler = get(layer.context, :sampler, Boltzmann.persistent_contdiv)
    _, hid, vis, _ = sampler(
        layer.rbm,
        reshape(X, length(X), 1),
        layer.context
    )

    return hid, vis
end
