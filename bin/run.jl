#using Logging
using DataFrames
import ArgParse: ArgParseSettings, @add_arg_table, parse_args
import YAML: load

include("../src/HippocampalModel.jl")

import HippocampalModel: BaseExperiment, execute, analyze


function run(experiments::Array; reps=1)
    timestamp = Libc.strftime("%Y-%m-%d_%H-%M-%S", time())
    directory = joinpath("data", "output", "$(timestamp)")
    mkdir(directory)

    datasets = Dict()

    for i in 1:length(experiments)
        exp_dict = experiments[i]
        exp_dict["directory"] = directory

        total_results = nothing

        exp = BaseExperiment(exp_dict)
        for i in 1:reps
            # To make sure we are using the exact same datasets
            # we reuse the same type instance
            if haskey(datasets, i)
                exp.dataset = datasets[i]
            else
                datasets[i] = exp.dataset
            end
            results = execute(exp)
            results["proactive"][:Rep] = fill(i, length(results["proactive"][1]))
            results["retroactive"][:Rep] = fill(i, length(results["retroactive"][1]))

            if total_results == nothing
                total_results = results
            else
                total_results["proactive"] = vcat(total_results["proactive"], results["proactive"])
                total_results["retroactive"] = vcat(total_results["retroactive"], results["retroactive"])
            end
            exp = BaseExperiment(exp_dict)
        end
        stats_df = analyze(exp, total_results)
        name = exp_dict["name"]
        writetable("$(directory)/$(name)_proactive_results.csv", total_results["proactive"])
        writetable("$(directory)/$(name)_retroactive_results.csv", total_results["retroactive"])
        writetable("$(directory)/$(name)_retroactive_summary_stats.csv", stats_df)
    end
end


function main()
    parse_settings = ArgParseSettings()

    @add_arg_table parse_settings begin
        "settings"
            help = "The path to the settings file. (should be in YAML"
            required = true
    end

    args = parse_args(parse_settings)
    run(
        load(
            open(args["settings"])
        ),
        reps=5
    )
end

main()
