module VeryDiff

using Zygote
using Zygote.ChainRulesCore
using MaskedArrays
using LinearAlgebra
using SparseArrays
using VNNLib
using NLSolvers

import Zygote.ChainRulesCore.rrule

using TimerOutputs
const to = TimerOutput()

include("Definitions.jl")
include("Util.jl")
include("Network.jl")
include("Zonotope.jl")
include("Layers_Zonotope.jl")
include("Layers_DiffZonotope.jl")
include("Verifier.jl")

export Network,GeminiNetwork,Layer,Dense,ReLU,WrappedReLU
export parse_network
export Zonotope, DiffZonotope, PropState
export zono_optimize
export verify_network

end # module AlphaZono
