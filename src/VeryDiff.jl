module VeryDiff

using Zygote
using Zygote.ChainRulesCore
using MaskedArrays
using LinearAlgebra
using SparseArrays
using VNNLib
using NLSolvers
#using ThreadPinning

# We have our own multithreadding so we don't want to use BLAS multithreadding
function __init__()
    BLAS.set_num_threads(1)
end

#pinthreads(:cores)


import Zygote.ChainRulesCore.rrule

using TimerOutputs
const to = TimerOutput()

#using Enzyme
#using NLSolvers

using JuMP
using GLPK
using Gurobi

const GRB_ENV = Ref{Any}(nothing)

function __init__()
    GRB_ENV[] = Gurobi.Env()
end

include("Definitions.jl")
include("Util.jl")
include("Network.jl")
include("Zonotope.jl")
include("Layers_Zonotope.jl")
include("Layers_DiffZonotope.jl")
include("MultiThreadding.jl")
include("Properties.jl")
include("Verifier.jl")

export Network,GeminiNetwork,Layer,Dense,ReLU,WrappedReLU
export parse_network
export Zonotope, DiffZonotope, PropState
export zono_optimize, zono_bounds
export verify_network
export get_epsilon_property, epsilon_split_heuristic
export get_top1_property, top1_configure_split_heuristic

end # module AlphaZono
