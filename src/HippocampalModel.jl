module HippocampalModel

VERSION < v"0.4-" && using Docile
using Compat
using DataFrames
using DataStructures
using Gadfly
using Bootstrap
using Boltzmann
using TrajectoryMatrices

import JSON
import Boltzmann: AbstractRBM

export Experiment, BaseExperiment, execute, analyze, GeneratedData


include("utils.jl")
include("datasets.jl")
include("models.jl")
include("evaluation.jl")
include("experiment.jl")

end
