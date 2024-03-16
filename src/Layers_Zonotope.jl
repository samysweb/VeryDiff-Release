using Zygote.ChainRulesCore

import VNNLib.NNLoader.Network
import VNNLib.NNLoader.Dense
import VNNLib.NNLoader.ReLU

import Zygote.ChainRulesCore.rrule

function (N::Network)(Z :: Zonotope, P :: PropState)
    return foldl((Z,L) -> L(Z,P),N.layers,init=Z)
end

function (L::Dense)(Z :: Zonotope,P :: PropState)
    #return @timeit to "Zonotope_DenseProp" begin
    G = L.W * Z.G
    c = L.W * Z.c .+ L.b
    return Zonotope(G,c)
    #end
end

function get_slope(l,u, alpha)
    if u <= 0
        return 0.0
    elseif l >= 0
        return 1.0
    else
        return alpha
    end
end

function (L::ReLU)(Z :: Zonotope, P :: PropState; bounds = nothing)
    #return @timeit to "Zonotope_ReLUProp" begin
    row_count = size(Z.G,1)
    if isnothing(bounds)
        bounds = zono_bounds(Z)
    end
    lower = @view bounds[:,1]
    upper = @view bounds[:,2]
    
    α = clamp.(upper./(upper.-lower),0.0,1.0)
    # Use is_onesided to compute 
    λ = ifelse.(upper.<=0.0,0.0,ifelse.(lower.>=0.0,1.0,α))

    crossing = lower.<0.0 .&& upper.>0.0

    γ = 0.5 .* max.(-λ .* lower,0.0,((-).(1.0,λ)).*upper)  # Computed offset (-λl/2)

    Ĝ = zeros(Float64,row_count, size(Z.G,2)+count(crossing))
    #zeros(row_count, size(Z.G,2)+count(crossing))
    Z.G .*= λ
    Ĝ[:,1:size(Z.G,2)] .= Z.G
    Ĝ[crossing,size(Z.G,2)+1:end] .=  (@view I(row_count)[crossing, crossing])
    Ĝ[:,size(Z.G,2)+1:end] .*= γ

    ĉ = λ .* Z.c .+ crossing.*γ

    return Zonotope(Ĝ, ĉ)
    #end
end
