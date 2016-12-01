@doc doc"""
    Normalizes a function to between 0 and 1 based on min and max of the input pattern.
""" ->
function normalize(pattern; min=0.0, max=1.0, epsilon=0.0001)
    pattern[indmax(pattern)] += epsilon

    if min < max
        if minimum(pattern) > maximum(pattern) || isnan(mean(pattern))
            println(pattern)
            error("pattern has NaNs in it for some reason")
        end

        range = maximum(pattern) - minimum(pattern)
        C = (max - min) / range

        scaled = scale(C, pattern)
        result = scaled + (min - minimum(scaled))

        #normalized = (pattern - minimum(pattern)) / (maximum(pattern) + epsilon)
        #result = (normalized + min) * (max - min)
        if isnan(mean(result))
            println(result)
            error("result has NaNs in it for some reason")
        end
    elseif min == max
        result = fill(min, size(pattern))
    else
        error("min > max")
    end

    return result
end

function makecontext(kwargs::Dict)
    context = Dict{Symbol, Any}()

    for (k, v) in kwargs
        if symbol(k) in Boltzmann.KNOWN_OPTIONS
            context[symbol(k)] = v
        end
    end

    return context
end

for i in [ "convert", "monitoring", "plotting", "stats" ]
    include(joinpath("utils", "$(i).jl"))
end
