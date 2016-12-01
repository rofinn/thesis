@doc doc"""
    All Evaluators should have a train and test method that takes the model and
""" ->
abstract Evaluator
const SAVE_FILE = "/tmp/model.hdf5"

train!(self::Evaluator, model, data::AbstractDataset) = error("train! has not been implemented for type $(typeof(self)).")
test(self::Evaluator, model, data::AbstractDataset) = error("test has not been implemented for type $(typeof(self)).")
#test(eval::Evaluator, model, prep::Preprocessor, postp::Postprocessor) = error("test has not been implemented for type $(typeof(self)).")

for i in ("generative",)
    include(joinpath("evaluation", "$(i).jl"))
end

@compat const EVALUATORS = Dict(
    # "mocha" => MochaEvaluator,
    "generative" => GenerativeEvaluator
)

function EvaluatorFactory(kwargs::Dict)
    evaluator = EVALUATORS[kwargs["type"]]
    evaluator(kwargs)
end

"""
`TestMonitor` collects test statistics

Fields:

`percent_matches` - simply store the percent matches from test
`hidden_summary` - keep track of average percent match relative to
    hidden activation in a dictionary
    ```
    {
        <label>: {
            "summary": tuple(avg_percent_match, avg_hid_act),
            "data": Array{tuple(percent_match, hid_act)}
        )
        ...
    }
    ```
    the summary contains a sum until `commit!` is called. There is probably a
    better approach, but it is probably best to keep the changes minimal at this point.
"""
type TestMonitor <: Monitor
    percent_matches::Array
    hidden_summary::Dict

    function TestMonitor()
        new(
            Array(Float64, 0),
            Dict()
        )
    end
end

function update!(m::TestMonitor, label, expected::Array, vis::Array, hid::Array)
    percent_match = sum(vis .== expected) / length(vis)
    push!(m.percent_matches, percent_match)

    hid = convert(Array{Int}, round(hid))
    if haskey(m.hidden_summary, label)
        m.hidden_summary[label]["summary"] = (
            m.hidden_summary[label]["summary"][1] + percent_match,
            m.hidden_summary[label]["summary"][2] + hid
        )
    else
        m.hidden_summary[label] = Dict(
            "summary" => (percent_match, hid),
            "data" => Tuple[]
        )
    end

    push!(m.hidden_summary[label]["data"], (percent_match, hid))
end

function commit!(m::TestMonitor)
    for (key, value) in m.hidden_summary
        value["summary"] = (
            value["summary"][1] / length(value["data"]),
            round(Int, (value["summary"][2] / length(value["data"])))
        )
    end
end

"""
Simply tests that model can reconstruct the pattern it is given.
"""
function test(model::AbstractModel, data::Data)
    X = data.X
    y = data.y
    monitor = TestMonitor()

    for i in 1:length(y)
        inputs = X[:, i]
        hid, vis, = generate(model, inputs)
        update!(monitor, y[i], inputs, vis, hid)
    end

    commit!(monitor)

    matches_mean = mean(monitor.percent_matches)
    matches_min = minimum(monitor.percent_matches)
    matches_max = maximum(monitor.percent_matches)
    matches_std = std(monitor.percent_matches)
    info("Mean: $matches_mean, Std: $matches_std, Min: $matches_min, Max: $matches_max")
    #return percent_matches
    return monitor.percent_matches, monitor.hidden_summary
end

"""
Tests how well a model is able to generalize an input pattern
to a parent class or category.
"""
function test(model::AbstractModel, prototypes::Data, data::Data)
    X = data.X
    y = data.y
    monitor = TestMonitor()

    # println(size(y))
    # println(size(X))
    for i in 1:length(y)
        inputs = X[:, i]
        prototype_idx = findfirst(prototypes.y, y[i])
        #println(size(prototypes.y))
        #println(size(prototypes.X))
        expected = prototypes.X[:, prototype_idx]
        hid, vis, = generate(model, inputs)
        update!(monitor, y[i], round(expected), round(vis), round(hid))
    end

    commit!(monitor)

    matches_mean = mean(monitor.percent_matches)
    matches_min = minimum(monitor.percent_matches)
    matches_max = maximum(monitor.percent_matches)
    matches_std = std(monitor.percent_matches)
    info("Mean: $matches_mean, Std: $matches_std, Min: $matches_min, Max: $matches_max")
    #return percent_matches
    return monitor.percent_matches, monitor.hidden_summary
end

function proactive_summary(proactive_hiddens)
    lbl = Float64[]
    itr = Int64[]
    acc = Float64[]
    for i in 1:length(proactive_hiddens)
        for (key, value) in proactive_hiddens[i]
            for val in value["data"]
                push!(acc, val[1])
                push!(itr, i)
                push!(lbl, key)
            end
        end
    end

    return DataFrame(
        Group = itr,
        Label = lbl,
        Accuracy = acc
    )
end

"""
DataFrame Summary:
    Iteration | Label | Diff | Age | Active | Accuracy | mean overlap | max overlap
"""
function retroactive_summary(proactive_hiddens, retroactive_hiddens)
    lbl = Float64[]
    itr = Int64[]
    diff = Float64[]
    acc = Float64[]
    act = Float64[]
    mean_overlap = Float64[]
    max_overlap = Float64[]
    for (key, value) in retroactive_hiddens
        for i in 1:length(proactive_hiddens)
            if haskey(proactive_hiddens[i], key)
                l = length(value["data"])
                append!(itr, fill(i, l))
                append!(lbl, fill(key, l))
                iter_val = proactive_hiddens[i][key]

                for val in value["data"]
                    push!(act, mean(val[2]))
                    if length(val[2]) == length(iter_val["summary"][2])
                        push!(diff, 1 - (sum(val[2] .== iter_val["summary"][2]) / length(val[2])))
                    else
                        push!(diff, 0.0)
                    end

                    push!(acc, val[1])

                    overlap = Float64[]
                    for (k, v) in retroactive_hiddens
                        if k != key
                            this = val[2]
                            other = v["summary"][2]

                            if length(this) == length(other)
                                push!(overlap, sum( (this & other) .== 1 ) / length(this))
                            else
                                push!(overlap, 0.0)
                            end
                        else
                            push!(overlap, 0.0)
                        end
                    end
                    push!(mean_overlap, mean(overlap))
                    push!(max_overlap, maximum(overlap))
                end
                break
            end
        end
    end

    return DataFrame(
        Group = itr,
        Label = lbl,
        Diff = diff,
        #Age = age,
        Activity = act,
        Accuracy = acc,
        Overlap = mean_overlap,
    )
end
