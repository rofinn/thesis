import DataStructures: OrderedDict

abstract AbstractModel

age!(m::AbstractModel, args...; kwargs...) = nothing

for i in [ "layers" ]
    include(joinpath("models", "$(i).jl"))
end

@compat const MODELS = Dict(
    "base" => BaseRBM,
    "neurogenesis" => Neurogenesis,
    "conditional" => Conditional,
)

include(joinpath("models", "hippocampaldbn.jl"))
MODELS["hippocampaldbn"] = HippocampalDBN

@doc doc"""
    Takes an array of models kwargs and uses the 'type'
    key from each kwargs dict to determine which model should
    be created.

    The keys for the dict returned are either extracted from the model
    params or infered from the model type.
""" ->
function ModelsFactory(models::Array, directory, feature_length=-1; shared_kwargs=Dict())
    results = OrderedDict{ASCIIString, AbstractModel}()

    for kwargs in models
        constructor = MODELS[kwargs["type"]]
        name = if haskey(kwargs, "name")
            kwargs["name"]
        else
            "$(typeof(constructor))"
        end

        kwargs["directory"] = joinpath(directory, name)
        if feature_length > 0
            kwargs["visible"] = feature_length
        end

        # Call the constructor, but merge the shared kwargs in and
        # convert the keys to symbols
        model = constructor(
            Dict(
                [(Symbol(k), v) for (k, v) in merge(shared_kwargs, kwargs)]
            )
        )
        results[name] = model
    end

    return results
end
