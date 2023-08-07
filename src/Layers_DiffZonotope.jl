using Zygote.ChainRulesCore

import VNNLib.NNLoader.Network
import VNNLib.NNLoader.Dense
import VNNLib.NNLoader.ReLU

function propagate_diff_layer(Ls :: Tuple{Dense,Dense,Dense}, Z::DiffZonotope, P::PropState)
    return @timeit to "DiffZonotope_DenseProp" begin
    println("Dense")
    L1, ∂L, L2 = Ls
    ∂G = L1.W*Z.∂Z.G[:,:]
    input_dim = size(Z.Z₂,2)-Z.num_approx₂
    accessed_range = 1:input_dim
    if Z.num_approx₂ > 0
        approx_two_range = input_dim + Z.num_approx₁ + 1
        accessed_range = [accessed_range;approx_two_range:approx_two_range+Z.num_approx₂-1]
    end
    ∂G[:,accessed_range] .+= ∂L.W*Z.Z₂.G
    ∂c = L1.W*Z.∂Z.c .+ ∂L.W * Z.Z₂.c .+ ∂L.b
    ∂Z_new = Zonotope(∂G,∂c)
    return DiffZonotope(L1(Z.Z₁,P),L2(Z.Z₂,P),∂Z_new,Z.num_approx₁,Z.num_approx₂,Z.∂num_approx)
    end
end

function propagate_diff_layer(Ls :: Tuple{ReLU,ReLU,ReLU}, Z::DiffZonotope, P::PropState)
    return @timeit to "DiffZonotope_DenseProp" begin
    println("ReLU")
    L1, _, L2 = Ls
    bounds₁ = zono_bounds(Z.Z₁)
    bounds₂ = zono_bounds(Z.Z₂)
    ∂bounds = zono_bounds(Z.∂Z)
    input_dim = size(Z.Z₂,2)-Z.num_approx₂

    lower₁ = @view bounds₁[:,1]
    upper₁ = @view bounds₁[:,2]
    lower₂ = @view bounds₂[:,1]
    upper₂ = @view bounds₂[:,2]
    ∂lower = @view ∂bounds[:,1]
    ∂upper = @view ∂bounds[:,2]

    α = ∂upper ./ (∂upper - ∂lower)
    # 
    λ = ifelse.((upper₁.<=0.0 .&& upper₂ .<= 0.0) , 0.0, ifelse.(lower₁.>=0.0 .&& lower₂ .>= 0.0, 1.0, ifelse.(∂upper .<= 0.0,0.0,α)))

    crossing = (lower₁ .< 0.0 .&& upper₁ .> 0.0) .|| (lower₂ .< 0.0 .&& upper₂ .> 0.0)
    
    δ = 0.5*max.(∂upper,-∂lower) #ifelse.(∂upper>-∂lower, 0.5*∂upper, -0.5*∂lower)

    if any(∂upper .<= 0)
        @warn "Contains negative differental upper bound"
    end

    γ = crossing .* (-δ .- λ .*∂lower ) # Decrease by ∂lower to ensure 0 reachable everywhere; decrease by 0.5*∂upper for approximation (additional dimension scaled by 0.5*∂upper)

    ĉ = λ .* Z.∂Z.c + γ
    Ĝ = λ .* Z.∂Z.G


    input_dim = size(Z.Z₂,2)-Z.num_approx₂
    row_count = size(lower₁,1)
    println("Crossing: ",count(x->x,crossing),"/",row_count)
    # The 1e-5 margin is necessary, because we otherwise observed floaty rounding errors in the output intervals
    E = (δ.+ sign.(δ)*1e-5) .* I(row_count)[:, crossing]
    #∂Z_new = Zonotope([Ĝ E], ĉ)
    Z₁_new = L1(Z.Z₁,P;bounds = bounds₁)
    Z₂_new = L2(Z.Z₂,P;bounds = bounds₂)
    ∂Z_new = Zonotope(hcat(
        Ĝ[:,1:input_dim+Z.num_approx₁],
        zeros(Float64,(row_count,size(Z₁_new.G,2)-Z.num_approx₁-input_dim)),
        Ĝ[:,input_dim+Z.num_approx₁+1:input_dim+Z.num_approx₁+Z.num_approx₂],
        zeros(Float64,(row_count,size(Z₂_new.G,2)-Z.num_approx₂-input_dim)),
        Ĝ[:,input_dim+Z.num_approx₁+Z.num_approx₂+1:end],
        E
    ), ĉ)
    # i=1
    # if size(∂lower,1)>=i
    #     println(lower₁[i])
    #     println(upper₁[i])
    #     println(lower₂[i])
    #     println(upper₂[i])
    #     println(∂lower[i])
    #     println(∂upper[i])
    #     println(crossing[i])
    #     println(λ[i])
    #     println(δ[i])
    #     println(γ[i])
    #     println(∂Z_new.c[i])
    #     println(∂Z_new.G[i,:])
    # end
    return DiffZonotope(Z₁_new,Z₂_new, ∂Z_new,size(Z₁_new,2)-input_dim,size(Z₂_new,2)-input_dim,Z.∂num_approx+size(E,2))
    end
end


function (N::GeminiNetwork)(Z :: DiffZonotope, P :: PropState)
    return foldl((Z,Ls) -> propagate_diff_layer(Ls,Z,P),zip(N.network1.layers,N.diff_network.layers,N.network2.layers),init=Z)
end