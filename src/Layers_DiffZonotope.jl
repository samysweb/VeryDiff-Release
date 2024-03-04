using Zygote.ChainRulesCore

import VNNLib.NNLoader.Network
import VNNLib.NNLoader.Dense
import VNNLib.NNLoader.ReLU

function propagate_diff_layer(Ls :: Tuple{Dense,Dense,Dense}, Z::DiffZonotope, P::PropState)
    #println("Prop dense")
    #return @timeit to "DiffZonotope_DenseProp" begin
    #println("Dense")
    L1, ∂L, L2 = Ls
    ∂G = Matrix{Float64}(undef, size(L1.W,1), size(Z.∂Z.G,2))
    mul!(∂G, L1.W, Z.∂Z.G)
    #∂G = L1.W*Z.∂Z.G
    input_dim = size(Z.Z₂,2)-Z.num_approx₂
    # accessed_range = 1:input_dim
    # if Z.num_approx₂ > 0
    #     approx_two_range = input_dim + Z.num_approx₁ + 1
    #     accessed_range = [accessed_range;approx_two_range:approx_two_range+Z.num_approx₂-1]
    # end
    #∂G[:,accessed_range] .+= ∂L.W*Z.Z₂.G
    mul!((@view ∂G[:,1:input_dim]), ∂L.W, (@view Z.Z₂.G[:,1:input_dim]), 1.0, 1.0)
    if Z.num_approx₂ > 0
        range_start = input_dim + Z.num_approx₁ + 1
        range_end = input_dim + Z.num_approx₁ + Z.num_approx₂
        mul!((@view ∂G[:,range_start:range_end]), ∂L.W, (@view Z.Z₂.G[:,(input_dim+1):end]), 1.0, 1.0)
    end
    #mul!((@view ∂G[:,accessed_range]), ∂L.W, Z.Z₂.G, 1.0, 1.0)
    #∂c = L1.W*Z.∂Z.c .+ ∂L.W * Z.Z₂.c .+ ∂L.b
    ∂c = L1.W*Z.∂Z.c #.+ ∂L.W * Z.Z₂.c .+ ∂L.b
    mul!(∂c, ∂L.W, Z.Z₂.c, 1.0, 1.0)
    ∂c .+= ∂L.b
    ∂Z_new = Zonotope(∂G,∂c)
    return DiffZonotope(L1(Z.Z₁,P),L2(Z.Z₂,P),∂Z_new,Z.num_approx₁,Z.num_approx₂,Z.∂num_approx)
    #end
end

function two_generator_bound(G::Matrix{Float64}, b, H::Matrix{Float64})
    @assert size(G,1) == size(H,1) && size(G,2) == size(H,2)
    return [sum(j->abs(G[i,j]+b*H[i,j]),1:size(G,2)) for i in 1:size(G,1)]
end

function propagate_diff_layer(Ls :: Tuple{ReLU,ReLU,ReLU}, Z::DiffZonotope, P::PropState)
    #println("Prop relu")
    #return @timeit to "DiffZonotope_DenseProp" begin
    #println("ReLU")
    input_dim = size(Z.Z₂,2)-Z.num_approx₂
    output_dim = size(Z.Z₂,1)

    L1, _, L2 = Ls

    offset_∂approx = input_dim + Z.num_approx₁ + Z.num_approx₂ + 1

    bounds₁₁ = zono_bounds(Z.Z₁)
    # Compute alternate version of bounds for Z₁ using Z₂ and ∂Z
    b1 = two_generator_bound(Z.Z₂.G[:,1:input_dim], 1.0, Z.∂Z.G[:,1:input_dim])
    if Z.num_approx₂ > 0
        range_start = input_dim + Z.num_approx₁ + 1
        range_end = input_dim + Z.num_approx₁ + Z.num_approx₂
        b1 .+= two_generator_bound(Z.Z₂.G[:,(input_dim+1):end], 1.0, Z.∂Z.G[:,range_start:range_end])
    end
    if Z.∂num_approx > 0
        b1 .+= sum(abs, Z.∂Z.G[:,offset_∂approx:end], dims=2)
    end
    bounds₁₂ = [(Z.Z₂.c + Z.∂Z.c).-b1 b1.+(Z.Z₂.c + Z.∂Z.c)]
    bounds₁ = bounds₁₁
    #bounds₁ = [max.(bounds₁₁[:,1],bounds₁₂[:,1]) min.(bounds₁₁[:,2],bounds₁₂[:,2])]

    b2 = two_generator_bound(Z.Z₁.G[:,1:end], -1.0, Z.∂Z.G[:,1:(input_dim+Z.num_approx₁)])
    if Z.∂num_approx > 0
        b2 .+= sum(abs, Z.∂Z.G[:,offset_∂approx:end], dims=2)
    end
    bounds₂₁ = [(Z.Z₁.c - Z.∂Z.c).-b2 b2.+(Z.Z₁.c - Z.∂Z.c)]
    bounds₂₂ = zono_bounds(Z.Z₂)
    bounds₂ = bounds₂₂
    #bounds₂ = [max.(bounds₂₁[:,1],bounds₂₂[:,1]) min.(bounds₂₁[:,2],bounds₂₂[:,2])]
    ∂bounds = zono_bounds(Z.∂Z)
    #println("Prop relu 1")
    lower₁ = @view bounds₁[:,1]
    upper₁ = @view bounds₁[:,2]
    lower₂ = @view bounds₂[:,1]
    upper₂ = @view bounds₂[:,2]
    ∂lower = @view ∂bounds[:,1]
    ∂upper = @view ∂bounds[:,2]

    α = ∂upper ./ (∂upper - ∂lower)
    # 
    #println("Prop relu 3.1")
    λ = ifelse.((upper₁.<=0.0 .&& upper₂ .<= 0.0) , 0.0, ifelse.(lower₁.>=0.0 .&& lower₂ .>= 0.0, 1.0, ifelse.(∂upper .<= 0.0,0.0,α)))
    #println("Prop relu 3.2")
    #lower₁_less0 = lower₁ .< 0.0
    #lower₂_less0 = lower₂ .< 0.0
    #upper₁_less0 = upper₁ .< 0.0
    #upper₂_less0 = upper₂ .< 0.0

    #crossing = (lower₁_less0 .&& (!).(upper₁_less0)) .|| (lower₂_less0 .&& (!).(upper₂_less0)) .|| ((!).(lower₁_less0) .&& upper₂_less0) .|| ((!).(lower₂_less0) .&& upper₁_less0)
    crossing = [
        ((l1 < 0.0 && u1 > 0) || (l2 < 0 && u2 > 0) || (l1 > 0 && u2 < 0) || (l2 > 0 && u1 < 0))
        for (l1, u1, l2, u2) in zip(lower₁, upper₁, lower₂, upper₂)
    ]
    #println("Prop relu 4")
    # Z₂ = Z₁ - ∂Z <-> Z₂ - Z₁ = ∂Z
    # TODO(steuber): We can do better than this if we consider the cases where only one network is instable
    # Difference between N1 - N2
    # Case 0: Both instable
    # Case 1: N1 zero -> Δ = 0-max(0,x1 - Δᵢ) = 0 or Δᵢ - x1 >= Δᵢ
    δ = 0.5 .* max.(∂upper,-∂lower) #ifelse.(∂upper>-∂lower, 0.5*∂upper, -0.5*∂lower

    γ = crossing .* (-δ .- λ .*∂lower ) # Decrease by ∂lower to ensure 0 reachable everywhere; decrease by 0.5*δ for approximation (additional dimension scaled by larger deviation)
    #println("Prop relu 5")
    ĉ = λ .* Z.∂Z.c + γ
    # Ĝ = λ .* Z.∂Z.G
    #println("Prop relu 6")
    row_count = size(lower₁,1)
    num_crossing = count(crossing)
 
    # TODO(steuber): This seems like a bad idea?
    # E = (δ.+ sign.(δ)*1e-5) .* (@view I(row_count)[:, crossing])
    #∂Z_new = Zonotope([Ĝ E], ĉ)
    Z₁_new = L1(Z.Z₁,P;bounds = bounds₁)
    Z₂_new = L2(Z.Z₂,P;bounds = bounds₂)
    #println("Prop relu 8")
    Ĝ = Matrix{Float64}(undef,row_count, size(Z₁_new.G,2)+(size(Z₂_new.G,2)-input_dim)+Z.∂num_approx+num_crossing)
    # Input space columns + approx columns of net 1
    num_cols = (input_dim+Z.num_approx₁)
    Ĝ[:,1:num_cols] .= λ .* (@view Z.∂Z.G[:,1:num_cols])
    #Ĝ[:,1:num_cols] .*= λ
    Ĝ[:,num_cols+1:size(Z₁_new.G,2)] .= 0.0

    # Approx columns of net 2
    if Z.num_approx₂ > 0
        offset_cols = size(Z₁_new.G,2)+1
        offset_cols_z = input_dim+Z.num_approx₁+1
        num_cols = Z.num_approx₂-1
        Ĝ[:,offset_cols:(offset_cols+num_cols)] .= λ .* (@view Z.∂Z.G[:,offset_cols_z:(offset_cols_z+num_cols)])
        #Ĝ[:,offset_cols:(offset_cols+num_cols)] .*= λ
    end
    range_start = size(Z₁_new.G,2)+Z.num_approx₂+1
    range_end = size(Z₁_new.G,2)+size(Z₂_new.G,2)-input_dim
    Ĝ[:,range_start:range_end] .= 0.0

    # Approx columns of ∂
    if Z.∂num_approx > 0
        offset_cols = size(Z₁_new.G,2)+size(Z₂_new.G,2)-input_dim+1
        offset_cols_z = input_dim+Z.num_approx₁+Z.num_approx₂+1
        num_cols = Z.∂num_approx-1
        Ĝ[:,offset_cols:(offset_cols+num_cols)] .= λ .* (@view Z.∂Z.G[:,offset_cols_z:end])
        #Ĝ[:,offset_cols:(offset_cols+num_cols)] .*= λ
    end

    # New Approx columns
    offset_cols = size(Z₁_new.G,2)+size(Z₂_new.G,2)-input_dim+Z.∂num_approx+1
    Ĝ[:,offset_cols:end] .= (δ.+ sign.(δ)*1e-5) .* (@view I(row_count)[:, crossing])

    ∂Z_new = Zonotope(Ĝ, ĉ)

    # ∂Z_new = Zonotope(hcat(
    #     (@view Ĝ[:,1:input_dim+Z.num_approx₁]),
    #     zeros(Float64,(row_count,size(Z₁_new.G,2)-Z.num_approx₁-input_dim)),
    #     (@view Ĝ[:,input_dim+Z.num_approx₁+1:input_dim+Z.num_approx₁+Z.num_approx₂]),
    #     zeros(Float64,(row_count,size(Z₂_new.G,2)-Z.num_approx₂-input_dim)),
    #     (@view Ĝ[:,input_dim+Z.num_approx₁+Z.num_approx₂+1:end]),
    #     E
    # ), ĉ)
    #println("Prop relu 9")
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
    return DiffZonotope(Z₁_new,Z₂_new, ∂Z_new,size(Z₁_new,2)-input_dim,size(Z₂_new,2)-input_dim,Z.∂num_approx+num_crossing)
    #end
end


function (N::GeminiNetwork)(Z :: DiffZonotope, P :: PropState)
    #println("Prop network")
    return foldl((Z,Ls) -> propagate_diff_layer(Ls,Z,P),zip(N.network1.layers,N.diff_network.layers,N.network2.layers),init=Z)
end