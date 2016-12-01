#=
The following file contains various functions for
manipulating the neurogenesis models.
=#

using Boltzmann

const SNG_STATS_FILENAME = "SparseNeurogenesisStats.csv"
const SNG_DEFAULT_LEARNING_RATE = 0.1
const SNG_AGE_SCORING_COEFF = 0.15
const SNG_DIFF_SCORING_COEFF = 0.65
const SNG_ACT_SCORING_COEEF = 0.2
const SNG_CONN_START = 0.3

type Neurogenesis <: AbstractLayer
    rbm::AbstractRBM
    context::Dict
    hidden_time::Array
    turnover::Float64
    growth_rate::Float64
    weight_decay_min::Float64
    weight_decay_max::Float64
    lr_min::Float64
    lr_max::Float64
    sparsity_cost_min::Float64
    sparsity_cost_max::Float64
    hidden_calc::Function
end

function Neurogenesis(kwargs::Dict; input_units=Bernoulli)
    nvis = kwargs[:visible]
    nhid = kwargs[:hidden]

    rbm = get(
        kwargs,
        :rbm,
        RBM(Float64, input_units, Bernoulli, nvis, nhid; sigma=0.001)
    )

    hidden_time = rand(nhid)
    sparse_conn = get(kwargs, :sparse_conn, false)
    hidden_calc =
        if haskey(kwargs, :hidden_calc) && kwargs[:hidden_calc] == "regulated"
            regulated()
        else
            replacement
        end

    context = makecontext(kwargs)
    context[:weight_decay_kind] = get(kwargs, :weight_decay_kind, :l1)
    context[:weight_decay_rates] = zeros(nhid)
    context[:lrs] = zeros(nhid)
    context[:sparsity_target] = get(kwargs, :sparsity_target, 1 / nvis)
    context[:sparsity_costs] = zeros(nhid)
    context[:update] = get(kwargs, :update, update_h!)
    if sparse_conn
        context[:connections] = zeros(nhid, nvis)
    end

    result = Neurogenesis(
        rbm, context, hidden_time,
        get(kwargs, :turnover, 0.1),
        get(kwargs, :growth_rate, 0.05),
        get(kwargs, :weight_decay_min, 0.0),
        get(kwargs, :weight_decay_max, 0.005),
        get(kwargs, :lr_min, 0.0025),
        get(kwargs, :lr_max, 0.1),
        get(kwargs, :sparsity_cost_min, 0.0),
        get(kwargs, :sparsity_cost_max, 0.1),
        hidden_calc
    )

    update_params!(result)
    update_connections!(result)

    return result
end

#####################################################
# Function to help with maintaining the neurogenesis
# parameters. Mostly to ensure that change to parameters
# and calculations remain consistent and organized.
#######################################################
function update_params!(model::Neurogenesis)
    growths = gompertz(model.hidden_time)
    model.context[:weight_decay_rates] = normalize(
        1 - growths,
        min=model.weight_decay_min,
        max=model.weight_decay_max
    )

    model.context[:lrs] = normalize(
        1 - growths,
        min=model.lr_min,
        max=model.lr_max
    )

    model.context[:sparsity_costs] = normalize(
        growths,
        min=model.sparsity_cost_min,
        max=model.sparsity_cost_max
    )
end

####################################################
# Functions specific to imposing sparse connectivity
####################################################
function update_connections!(model::Neurogenesis)
    if haskey(model.context, :connections)
        growths = gompertz(model.hidden_time)
        C = model.context[:connections]
        num_conn = normalize(
            growths,
            min=(length(model.rbm.vbias) * SNG_CONN_START),
            max=length(model.rbm.vbias)
        )

        for i in 1:length(num_conn)
            increase = num_conn[i] - sum(C[i, :])
         #   println(increase)

            for j in 1:increase
                inactive = findin(C[i, :], 0.0)
                if length(inactive) > 0
                    index = rand(1:length(inactive))
                    C[i, inactive[index]] = 1.0
                end
            end
        end
    end
end

function ratio_connected(C::Array{Float64, 2})
    ratios = zeros(size(C, 1))
    for i in 1:length(ratios)
        ratios[i] = length(findin(C[i, :], 1.0)) / size(C, 2)
    end

    return ratios
end

#######################################################
# Functions to support updating weights when the hidden
# units have variable learning rates, weight decays and
# sparsity costs.
#######################################################
function update_h!{T}(rbm::AbstractRBM, X::Array{T,2}, dtheta::Tuple, ctx::Dict)
    _dtheta = copy(dtheta)

    # apply gradient updaters. note, that updaters all have
    # the same signature and are thus composable
    _dtheta = grad_apply_learning_rates!(rbm, X, _dtheta, ctx)
    # grad_apply_momentum!(rbm, X, _dtheta, ctx)
    # _dtheta = grad_apply_weight_decays!(rbm, X, _dtheta, ctx)
    _dtheta = grad_apply_sparsities!(rbm, X, _dtheta, ctx)
    # add gradient to the weight matrix

    if haskey(ctx, :connections)
        _dtheta = (_dtheta[1] .* ctx[:connections], _dtheta[2:end]...)
    end

    update_weights!(rbm, _dtheta, ctx)

    # Update the original
    for i in 1:length(dtheta)
        try
            map(
                j -> dtheta[i][j] = _dtheta[i][j],
                eachindex(dtheta[i])
            )
        catch
            println(dtheta)
            println(_dtheta)
        end
    end
end

function grad_apply_learning_rates!{T,V,H}(rbm::RBM{T,V,H}, X::Array{T,2},
                                          dtheta::Tuple, ctx::Dict)
    dW, db, dc = dtheta
    lrs = ctx[:lrs]

    # same as: dW *= lr

    try
        return (
            dW .* lrs,
            db * mean(lrs),
            dc .* lrs
        )
    catch
        println(size(dW))
        println(size(db))
        println(size(dc))
        println(size(lrs))
        rethrow()
    end
end

function grad_apply_learning_rates!{T,V,H}(rbm::ConditionalRBM{T,V,H}, X::Array{T,2},
                                           dtheta::Tuple, ctx::Dict)
    dW, dA, dB, db, dc = dtheta
    lrs = ctx[:lrs]

    # same as: dW *= lr
    try
        return (
            dW .* lrs,
            dA * mean(lrs),
            dB .* lrs,
            db * mean(lrs),
            dc .* lrs
        )
    catch
        println(size(dW))
        println(size(dA))
        println(size(dB))
        println(size(db))
        println(size(dc))
        println(size(lrs))
        rethrow()
    end
end

function grad_apply_weight_decays!{T,V,H}(rbm::RBM{T,V,H}, X::Array{T,2},
                                         dtheta::Tuple, ctx::Dict)
    # The decay penalty should drive all weights toward
    # zero by some small amount on each update.
    dW, db, dc = dtheta
    decay_kind = ctx[:weight_decay_kind]
    decay_rates = ctx[:weight_decay_rates]
    is_l2 = haskey(ctx, :l2)
    if decay_kind == :l2
        # same as: dW -= decay_rate * W
        dW -=  rbm.W .* decay_rates
    elseif decay_kind == :l1
        # same as: dW -= decay_rate * sign(W)
        dW -= sign(rbm.W) .* decay_rates
    end
    return (dW, db, dc)
end

function grad_apply_weight_decays!{T,V,H}(rbm::ConditionalRBM{T,V,H}, X::Array{T,2},
                                         dtheta::Tuple, ctx::Dict)
    # The decay penalty should drive all weights toward
    # zero by some small amount on each update.
    dW, dA, dB, db, dc = dtheta
    decay_kind = ctx[:weight_decay_kind]
    decay_rates = ctx[:weight_decay_rates]
    is_l2 = haskey(ctx, :l2)
    if decay_kind == :l2
        # same as: dW -= decay_rate * W
        dW -= rbm.W .* decay_rates
        dB -= rbm.B .* decay_rates
    elseif decay_kind == :l1
        # same as: dW -= decay_rate * sign(W)
        dW -= sign(rbm.W) .* decay_rates
        dB -= sign(rbm.B) .* decay_rates
    end
    return (dW, dA, dB, db, dc)
end

function grad_apply_sparsities!{T,V,H}(rbm::RBM{T,V,H}, X::Array{T,2},
                                         dtheta::Tuple, ctx::Dict)
    # The sparsity constraint should only drive the weights
    # down when the mean activation of hidden units is higher
    # than the expected (hence why it isn't squared or the abs())
    dW, db, dc = dtheta
    cost = ctx[:sparsity_costs]
    target = ctx[:sparsity_target]
    curr_sparsity = mean(hid_means(rbm, X))
    penalty = cost * (curr_sparsity - target)
    # println("Sparsity Penalty = $(mean(penalty))")
    return (
        dW .- penalty,
        db - mean(penalty),
        dc .- penalty
    )
end

function grad_apply_sparsities!{T,V,H}(rbm::ConditionalRBM{T,V,H}, X::Array{T,2},
                                         dtheta::Tuple, ctx::Dict)
    # The sparsity constraint should only drive the weights
    # down when the mean activation of hidden units is higher
    # than the expected (hence why it isn't squared or the abs())
    dW, dA, dB, db, dc = dtheta
    cost = ctx[:sparsity_costs]
    target = ctx[:sparsity_target]
    curr_sparsity = mean(hid_means(
        rbm,
        Boltzmann.split_vis(rbm, X)[1]
    ))
    penalty = cost * (curr_sparsity - target)
    # println("Sparsity Penalty = $(mean(penalty))")
    return (
        dW .- penalty,
        dA,
        dB .- penalty,
        db - mean(penalty),
        dc .- penalty
    )
end

###############################################
# Functions needed for neurogenesis aging
# regardless of turnover method.
###############################################

function age!(layer::Neurogenesis; score=NaN)
    rbm = layer.rbm
    ctx = layer.context
    hidden_time = layer.hidden_time
    growth_rate = layer.growth_rate
    turnover = layer.turnover
    hidden_calc = layer.hidden_calc
    h = length(rbm.hbias)

    for i in 1:h
        if hidden_time[i] < (1.0 - growth_rate)
            hidden_time[i] += growth_rate
        end
    end

    # This condition could probably be a function
    if !isnan(score)
        create, destroy = hidden_calc(score, length(rbm.hbias), turnover)
        println(create)
        println(destroy)

        rankings = apoptosis(rbm.W, ctx, layer.hidden_time)
        @assert length(rankings) == h

        reset = min(create, destroy)
        to_rem = destroy - reset
        to_add = create - reset

        # println("Reseting weights")
        # Reset weights existing weights
        for i in rankings[1:reset]
            rbm.W[i, :] = rand(Normal(0, 0.01), length(rbm.vbias))
            rbm.hbias[i] = 0.0
            hidden_time[i] = 0

            # Reset connection for new neurons
            if haskey(ctx, :connections)
                ctx[:connections][i, :] = fill(0, length(rbm.vbias))
            end
        end

        #println("Computing hidden units to keep")
        to_keep = setdiff(1:h, rankings[reset+1:reset+to_rem])

        rbm.W = vcat(
            rbm.W[to_keep, :],
            rand(
                Normal(0, 0.01),
                (to_add, length(rbm.vbias))
            )
        )

        if isa(rbm, ConditionalRBM)
            rbm.B = vcat(
                rbm.B[to_keep, :],
                rand(
                    Normal(0, 0.01),
                    (to_add, length(rbm.vbias))
                )
            )
        end

        #println("Adding extra hidden bias units")
        rbm.hbias = vcat(rbm.hbias[to_keep], zeros(to_add))
        hidden_time = vcat(hidden_time[to_keep], zeros(to_add))
    end

    layer.hidden_time = hidden_time
    update_params!(layer)
    update_connections!(layer)
end

function regulated()
    r = 0.0
    prev = NaN

    function wrapped(score, hsize, turnover)
        new_score = score * hsize
        # Default behaviour is to add some nodes
        # delta = round(Int, (hsize * turnover) / 2)
        # create_delta = delta
        # destroy_delta = delta
        create_delta = 0
        destroy_delta = 0

        prev_r = r

        if !isnan(prev)
            r = prev == 0.0 ? abs(new_score / nextfloat(prev)) : abs(new_score / prev)
        end

        #if abs((r - 1) / (prev_r - 1)) <= 1.0
            # coeff = clamp(r - 1, -1, 1)
            # println("Coeff: $coeff")
            # offset = round(Int, (hsize * turnover * coeff) / 2)
            # create_delta -= offset
            # destroy_delta += offset
            cratio = clamp(r, 0.0, 1.0)
            create_delta = round(Int, hsize * (turnover * (1.0 - cratio)))
            dratio = clamp(r - 1.0, 0.0, 1.0)
            destroy_delta = round(Int, hsize * (turnover * dratio))
        #end

        prev = new_score

        println("Likelihood: $score")
        println("Score: $new_score")
        println("Ratio: $r")
        println("Create: $create_delta")
        println("Destroy: $destroy_delta")
        println("Size: $(hsize + create_delta - destroy_delta)")
        return create_delta, destroy_delta
    end

    return wrapped
end

function replacement(score, hsize, turnover)
    num_regen = int(length(hsize) * turnover)

    return num_regen, num_regen
end

function apoptosis(W, ctx::Dict, ages)
    if any(isnan, W)
        error("Weights have NaNs in them")
    end

    neuron_ages = 1 - ages
    weight_diff = vec(std(W, 2))
    weight_act = vec(mean(abs(W), 2))

    # If we are working with a sparsely connected
    # model then we need to take the differentiation
    # and activity of only the connections that exist
    if haskey(ctx, :connections)
        connected = ratio_connected(ctx[:connections])

        for i in 1:size(W, 1)
            conn = getindex(W[i,:], find(W[i,:]))
            weight_diff[i] = vec(std(conn))
            weight_act[i] = abs(mean(conn))
        end
    end

    rankings = (
        DEFAULT_APOP_WEIGHTING[:age] * neuron_ages +
        DEFAULT_APOP_WEIGHTING[:diff] * normalize(weight_diff) +
        DEFAULT_APOP_WEIGHTING[:act] * normalize(weight_act)
    ) / (DEFAULT_APOP_WEIGHTING[:age] + DEFAULT_APOP_WEIGHTING[:diff] + DEFAULT_APOP_WEIGHTING[:act])
    # rankings = weight_diff + weight_act) / 2
    return sortperm(rankings)
end


