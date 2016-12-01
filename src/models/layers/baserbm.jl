using Distributions
using ArrayViews
using Base.LinAlg.BLAS

import Base.length
import Boltzmann:
    sample_hiddens,
    sample_visibles,
    free_energy,
    update_weights!,
    gibbs,
    score_samples,
    contdiv,
    persistent_contdiv,
    fit_batch!,
    fit,
    transform,
    generate

type BaseRBM <: AbstractLayer
    #rbm::OldRBM
    rbm::AbstractRBM
    context::Dict

    function BaseRBM(kwargs::Dict)
        nvis = kwargs[:visible]
        nhid = kwargs[:hidden]

        # rbm = BernoulliRBM(nvis, nhid; sigma=0.001)
        rbm = get(
            kwargs,
            :rbm,
            RBM(Float64, Bernoulli, Bernoulli, nvis, nhid; sigma=0.001)
        )
        context = makecontext(kwargs)

        #rbm = OldRBM(kwargs)
        new(rbm, context)
    end
end

"""
Our Temporal CA1/CA3 layer is represented by wrapping a conditional RBM.
A queue of Array{Float64} is used to represent the history during training.
"""
type Conditional
    rbm::ConditionalRBM
    context::Dict{Symbol, Any}
    steps::Int
end

function Conditional(kwargs::Dict)
    nvis = kwargs[:visible]
    nhid = kwargs[:hidden]
    ncond = kwargs[:cond]
    steps = get(kwargs, :steps, 1)

    rbm = ConditionalRBM(Float64, Bernoulli, Bernoulli,
                        nvis, nhid, (nvis * steps) + ncond; sigma=0.001)

    context = makecontext(kwargs)

    return Conditional(rbm, context, steps)
end
