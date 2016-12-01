
using Boltzmann
using Distributions
using DataFrames
using Gadfly
using OnlineStats

include("../src/HippocampalModel.jl")

import HippocampalModel:
    NullReporter,
    LayerMonitor,
    getcontext,
    BaseRBM,
    Neurogenesis,
    lineplot,
    update!,
    age!

const CONTEXT = Dict(
    :debug => false,
    :growth_rate => 0.1,
    :n_gibbs => 5,
    :momentum => 0.9,
    :turnover => 0.1,
    :weight_decay_max => 0.0,
    :sparsity_cost_max => 0.0,
    :lr_min => 0.1,
    :lr_max => 0.1,
    # :update => Boltzmann.update_classic!,
    :terminator => [
        Boltzmann.EpochTerminator(100),
        Boltzmann.ConvergenceTerminator(),
    ]
)

function hidden_vs_time(X)
    ctx = copy(CONTEXT)

    monitor = LayerMonitor()

    for h in (5:1:100)
        ctx[:visible] = 100
        ctx[:hidden] = h
        m = BaseRBM(ctx)
        fit(m.rbm, X, m.context)
        update!(monitor, m, X)
    end

    times = monitor.runtimes
    likelihoods = monitor.scores
    nhidden = monitor.hsize

    times = (times + abs(minimum(times))) / (maximum(times) - minimum(times))
    likelihoods = (likelihoods + abs(minimum(likelihoods))) / (maximum(likelihoods) - minimum(likelihoods))

    return DataFrame(
        Size = vcat(nhidden, nhidden),
        Value = vcat(likelihoods, times),
        Type = vcat(fill("pseudo likelihood", length(times)), fill("time", length(times)))
    )
    return df
    # plot(df, x="Size", y="Value", color="Type", Geom.line)
    # plot(nhidden, Vector[likelihoods, times], axis=[:l :r], ylabel="pseudo likelihood",yrightlabel="time")
end

function analyze_convergence()
    ctx = getcontext(["pseudo likelihood", "time"])
    plots = Array{Plot}(2,2)

    plots[1,1] = lineplot(
        hidden_vs_time(Boltzmann.generate_dataset(Float64, 100; n_classes=10, n_obs=1000)),
        x="Size", y="Value", color="Type", Guide.title("Size vs Performance with 10 classes")
    )

    plots[1,2] = lineplot(
        hidden_vs_time(Boltzmann.generate_dataset(Float64, 100; n_classes=20, n_obs=1000)),
        x="Size", y="Value", color="Type", Guide.title("Size vs Performance with 20 classes")
    )

    plots[2,1] = lineplot(
        hidden_vs_time(Boltzmann.generate_dataset(Float64, 100; n_classes=30, n_obs=1000)),
        x="Size", y="Value", color="Type", Guide.title("Size vs Performance with 30 classes")
    )

    plots[2,2] = lineplot(
        hidden_vs_time(Boltzmann.generate_dataset(Float64, 100; n_classes=40, n_obs=1000)),
        x="Size", y="Value", color="Type", Guide.title("Size vs Performance with 40 classes")
    )

    write(plots, "../data/output/size_vs_performance.pdf")
end


function regulated_ng(classes::Array, start_size=150, ngroups=3, reps=100)
    results = DataFrame(LayerMonitor())

    for c in classes
        ctx = deepcopy(CONTEXT)
        ctx[:visible] = 100
        ctx[:hidden] = start_size
        ctx[:hidden_calc] = "regulated"
        ctx[:batch_size] = c * 100
        monitor = LayerMonitor(5)

        m = Neurogenesis(ctx; input_units=Normal)
        for g in 1:ngroups
            X = Boltzmann.generate_dataset(Float64, 100; n_classes=c, n_obs=1000)
            for i in 1:reps
                fit(m.rbm, X, m.context)
                iteration, runtime, score = update!(monitor, m, X)
                age!(m, score=score)
            end
        end
        result = DataFrame(monitor)
        result[:Classes] = fill(string(c), ngroups * reps)
        results = vcat(results, result)
    end
    return results
end

function analyze_regulated(;start_size=150, ngroups=3, classes=[10, 20, 30], reps=300)
    ctx = getcontext(map(x -> string(x), classes))
    df = regulated_ng(classes, start_size, ngroups, reps)
    println(df)

    p = lineplot(
        df, x="Iteration", y="Size", color="Classes",
        Guide.title("Size vs Iteration (regulated multigroup training)")
    )
    write(p, "../data/output/size_vs_iteration.pdf")
end

# analyze_convergence()
analyze_regulated(;start_size=150, ngroups=3, classes=[5, 10, 15], reps=100)
#analyze_regulated(; start_size=150, ngroups=1, classes=[5, 10, 15], reps=20)
