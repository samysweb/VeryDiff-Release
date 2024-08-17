import VNNLib.NNLoader.Network
import VNNLib.NNLoader.Dense
import VNNLib.NNLoader.ReLU

function (N::Network)(Z :: Zonotope, P :: PropState)
    return foldl((Z,L) -> L(Z,P),N.layers,init=Z)
end

function (L::Dense)(Z :: Zonotope,P :: PropState)
    return @timeit to "Zonotope_DenseProp" begin
    G = L.W * Z.G
    c = L.W * Z.c .+ L.b
    return Zonotope(G,c, Z.influence)
    end
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
    return @timeit to "Zonotope_ReLUProp" begin
    @timeit to "Bounds" begin
    row_count = size(Z.G,1)
    if isnothing(bounds)
        bounds = zono_bounds(Z)
    end
    lower = @view bounds[:,1]
    upper = @view bounds[:,2]
    end

    @timeit to "Vectors" begin
    α = clamp.(upper./(upper.-lower),0.0,1.0)
    # Use is_onesided to compute 
    λ = ifelse.(upper.<=0.0,0.0,ifelse.(lower.>=0.0,1.0,α))

    crossing = lower.<0.0 .&& upper.>0.0
    
    γ = 0.5 .* max.(-λ .* lower,0.0,((-).(1.0,λ)).*upper)  # Computed offset (-λl/2)

    ĉ = λ .* Z.c .+ crossing.*γ
    end
    
    @timeit to "Influence Matrix" begin
    if NEW_HEURISTIC
        # TODO(steuber): Can we avoid this reallocation?
        @timeit to "Allocation" begin
        #println(size(Z.influence,1), size(Z.influence,2)+count(crossing))
        influence_new = zeros(Float64, size(Z.influence,1), size(Z.influence,2)+count(crossing))
        end
        @timeit to "Set Matrix" begin
        influence_new[:,1:size(Z.influence,2)] .= Z.influence
        end
        # print("Hello")
        # print(size(influence_new))
        # print(size(Z.influence * Z.G[crossing,:]'))
        @timeit to "Multiply" begin
        influence_new[:,(size(Z.influence,2)+1):end] .=  abs.(Z.influence) * abs.(@view Z.G[crossing,:])'
        end
        # foreach(normalize!, eachcol(@view influence_new[:,(size(Z.influence,2)+1):end]))
    else
        influence_new = Z.influence
    end
    end

    @timeit to "Allocation" begin
    Ĝ = zeros(Float64,row_count, size(Z.G,2)+count(crossing))
    end
    #zeros(row_count, size(Z.G,2)+count(crossing))
    #Z.G .*= λ
    @timeit to "Set Matrix" begin
    Ĝ[:,1:size(Z.G,2)] .= Z.G
    Ĝ[crossing,size(Z.G,2)+1:end] .=  (@view I(row_count)[crossing, crossing])
    end
    @timeit to "Column Multiply" begin
    Ĝ[:,1:size(Z.G,2)] .*= λ
    Ĝ[:,size(Z.G,2)+1:end] .*= abs.(γ)
    end

    return Zonotope(Ĝ, ĉ, influence_new)
    end
end
