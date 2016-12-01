function bootstrap(df::DataFrame; ind=:Model, dep=:Accuracy, by=mean)
    model_diff(x::AbstractArray{Float64, 2}) = mean(x[:,2]) - mean(x[:,1])

    inds = unique(df[ind])
    combos = combinations(inds, 2)

    comparisons = ASCIIString[]
    avg_mean1 = Float64[]
    avg_mean2 = Float64[]
    t0s = Float64[]
    lower = Float64[]
    upper = Float64[]
    level = Float64[]
    resamples = Int[]
    biases = Float64[]
    standard_error = Float64[]

    for c in combos
        df1 = df[df[ind] .== c[1], :]
        df2 = df[df[ind] .== c[2], :]
        a1 = df1[dep]
        a2 = df2[dep]

        if haskey(df, :Rep)
            df1 = aggregate(df1[[:Rep, dep]], [:Rep], [by])
            df2 = aggregate(df2[[:Rep, dep]], [:Rep], [by])
            a1 = df1[symbol("$(dep)_mean")]
            a2 = df2[symbol("$(dep)_mean")]
        end

        a = hcat(a1, a2)

        bs = boot_balanced(a, model_diff, 10000)
        ci = ci_basic(bs, 0.99)

        push!(comparisons, "$(c[1]) vs $(c[2])")
        push!(avg_mean1, mean(a1))
        push!(avg_mean2, mean(a2))
        push!(t0s, ci.t0)
        push!(lower, ci.lower)
        push!(upper, ci.upper)
        push!(level, ci.level)
        push!(resamples, bs.m)
        push!(biases, bias(bs))
        push!(standard_error, se(bs))
    end

    return DataFrame(
        Comparison = comparisons,
        AvgMean1 = avg_mean1,
        AvgMean2 = avg_mean2,
        t0 = t0s,
        LowerBound = lower,
        UpperBound = upper,
        Level = level,
        Resamples = resamples,
        Bias = biases,
        StandardError = standard_error,
    )
end
