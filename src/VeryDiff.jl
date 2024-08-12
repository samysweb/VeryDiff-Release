module VeryDiff

#using MaskedArrays
using LinearAlgebra
#using SparseArrays
using VNNLib
#using ThreadPinning

using GLPK

NEW_HEURISTIC = true
USE_GUROBI = true

USE_DIFFZONO = true

# We have our own multithreadding so we don't want to use BLAS multithreadding
function __init__()
    BLAS.set_num_threads(1)
    GRB_ENV[] = Gurobi.Env()
    GRBsetintparam(GRB_ENV[], "OutputFlag", 0)
    GRBsetintparam(GRB_ENV[], "LogToConsole", 0)
    GRBsetintparam(GRB_ENV[], "Threads", 0)
    #GRBsetintparam(GRB_ENV[], "Method", 2)
    #       mnist_19_local_21.vnnlib        mnist_18_local_18
    #0 :    0.018826400587219343s/loop      0.03304489948205128s/loop
    #1 :    0.01705984154058722s/loop       0.03352098044717949s/loop
    #2 :    0.020955224224525042s/loop      0.038390683782564106s/loop
end

#pinthreads(:cores)

FIRST_ROUND = true

using TimerOutputs
const to = TimerOutput()

using JuMP
#using GLPK
using Gurobi

const GRB_ENV = Ref{Any}(nothing)

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
export get_epsilon_property, epsilon_split_heuristic, get_epsilon_property_naive
export get_top1_property, top1_configure_split_heuristic

end # module AlphaZono
