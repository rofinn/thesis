using DataFrames

include("../src/HippocampalModel.jl")

import HippocampalModel: BaseExperiment, execute, analyze
import YAML: load

settings = load(open("settings/frontiers-2015.yml"))
directory = "../HippocampalModel\ 2.jl/experiments/2015-08-29_14-46-46/"

for i in 1:length(settings)
   exp_dict = settings[i]
   exp_dict["directory"] = directory
   exp = BaseExperiment(exp_dict)
   name = exp_dict["name"]
   total_results = Dict(
       "proactive" => readtable("$(directory)/$(name)_proactive_results.csv"),
       "retroactive" => readtable("$(directory)/$(name)_retroactive_results.csv")
   )

   total_results["proactive"][:Model] = map(x -> unescape_string(x), total_results["proactive"][:Model])
   total_results["retroactive"][:Model] = map(x -> unescape_string(x), total_results["retroactive"][:Model])

   analyze(exp, total_results)
end
