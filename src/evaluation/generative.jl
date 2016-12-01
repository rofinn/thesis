import Boltzmann


@doc doc"""
    Simply runs the generative RBM with binary labels added
    to the input space. Testing is performed by selecting the class/label
    that minimizes the energy of model for the feature vector.
""" ->
type GenerativeEvaluator <: Evaluator
    sorted::Bool
    noise_iter::Int64
    age::Bool
    completion::Bool

    function GenerativeEvaluator(kwargs::Dict)
        gibb_steps = haskey(kwargs, "gibb_steps") ? kwargs["gibb_steps"] : 1
        noise_iter = haskey(kwargs, "noise") ? kwargs["noise"] : 0
        age = haskey(kwargs, "age") ? kwargs["age"] : true
        completion = get(kwargs, "completion", false)
        new(
            kwargs["sorted"],
            noise_iter,
            age,
            completion
        )
    end
end

"""
Iterative train test loops.
"""
function iter_train!(self::GenerativeEvaluator, model, data::GeneratedData)
    iterative_hid = Dict[]

    # If this is an AbstractLayer just attach
    # the LayerMonitor, but if this is the DBN
    # attach a layer monitor to the neurogenesis DG.
    monitor = LayerMonitor()
    if isa(model, AbstractLayer)
        model.context[:reporter] = monitor
    elseif isa(model, HippocampalDBN)
        model.dg.context[:reporter] = monitor
    end

    # For each group
    for group in iterator(data)
        # println(dump(group))
        info("Training and testing group $(group.groups[1])...")
        X = traindata(group).X

        # Train on the label
        fit(model, X)

        # Test with the training data
        info(self.completion)
        percent_matches, hid_sum =
            if self.completion
                test(model, prototypes(group), traindata(group))
            else
                test(model, traindata(group))
            end

        score = isa(model, HippocampalDBN) ? update!(monitor, model.dg, X)[3] : update!(monitor, model, X)[3]
        push!(iterative_hid, hid_sum)

        if self.age
            isa(model, HippocampalDBN) ? age!(model.dg, score=score) : age!(model, score)
        end
    end

    for i in 1:self.noise_iter
        # Present a random set of noise
        fit(model, round(rand(size(traindata(data).X, 1), 10)))
        update!(monitor, model, traindata(data).X)

        if self.age
            age!(model)
        end
    end

    # Finally run the full test
    info("Summary Test")
    summary, hid_sum =
        if self.completion
            test(model, prototypes(data), traindata(data))
        else
            test(model, traindata(data))
        end

    return (
        DataFrame(monitor),
        proactive_summary(iterative_hid),
        retroactive_summary(iterative_hid, hid_sum)
    )
end

@doc doc"""
    Simply fits the dataset to the model.
""" ->
function train!(self::GenerativeEvaluator, model, data::GeneratedData)
    X = traindata(data).X
    y = traindata(data).y

    ngroups = length(data.groups)
    iter_size = int(floor(length(y) / ngroups))
    for i in 1:ngroups
        iter_batch = X[:, ((i-1)*iter_size + 1):min(i*iter_size, size(X, 2))]
        fit(model, iter_batch)
    end
end
