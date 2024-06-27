import VNNLib.NNLoader.Network
import VNNLib.NNLoader.Dense
import VNNLib.NNLoader.ReLU

function propagate_diff_layer(Ls :: Tuple{Dense,Dense,Dense}, Z::DiffZonotope, P::PropState)
    #println("Prop dense")
    return @timeit to "DiffZonotope_DenseProp" begin
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
    ∂Z_new = Zonotope(∂G,∂c,Z.∂Z.influence)
    return DiffZonotope(L1(Z.Z₁,P),L2(Z.Z₂,P),∂Z_new,Z.num_approx₁,Z.num_approx₂,Z.∂num_approx)
    end
end

function two_generator_bound(G::Matrix{Float64}, b, H::Matrix{Float64})
    @assert size(G,1) == size(H,1) && size(G,2) == size(H,2)
    return [sum(j->abs(G[i,j]+b*H[i,j]),1:size(G,2)) for i in 1:size(G,1)]
end

function propagate_diff_layer(Ls :: Tuple{ReLU,ReLU,ReLU}, Z::DiffZonotope, P::PropState)
    #println("Prop relu")
    return @timeit to "DiffZonotope_ReLUProp" begin
    #println("ReLU")

    L1, _, L2 = Ls
    input_dim = size(Z.Z₂,2)-Z.num_approx₂

    # Compute Bounds
    bounds₁ = zono_bounds(Z.Z₁)
    # Compute alternative version of bounds via Z.Z₂+Z.∂Z
    # bounds₁₂ = zeros(size(Z.Z₂.G,1))
    # for i in 1:input_dim
    #     bounds₁₂ .+= abs.(Z.Z₂.G[:,i].+Z.∂Z.G[:,i])
    # end
    # for i in 1:Z.num_approx₂
    #     bounds₁₂ .+= abs.(Z.Z₂.G[:,input_dim+i].+Z.∂Z.G[:,input_dim+Z.num_approx₁+i])
    # end
    # for i in 1:Z.∂num_approx
    #     bounds₁₂ .+= abs.(Z.∂Z.G[:,input_dim+Z.num_approx₁+Z.num_approx₂+i])
    # end
    # bounds₁[:,1] = max.(bounds₁[:,1],(Z.Z₂.c .+ Z.∂Z.c) .- bounds₁₂)
    # bounds₁[:,2] = min.(bounds₁[:,2],(Z.Z₂.c .+ Z.∂Z.c) .+ bounds₁₂)
    bounds₂ = zono_bounds(Z.Z₂)
    # bounds₂₁ = zeros(size(Z.Z₂.G,1))
    # for i in 1:(input_dim+Z.num_approx₁)
    #     bounds₂₁ .+= abs.(Z.Z₁.G[:,i].-Z.∂Z.G[:,i])
    # end
    # for i in 1:Z.∂num_approx
    #     bounds₂₁ .+= abs.(Z.∂Z.G[:,input_dim+Z.num_approx₁+Z.num_approx₂+i])
    # end
    # bounds₂[:,1] = max.(bounds₂[:,1],(Z.Z₁.c .- Z.∂Z.c) .- bounds₂₁)
    # bounds₂[:,2] = min.(bounds₂[:,2],(Z.Z₁.c .- Z.∂Z.c) .+ bounds₂₁)
    ∂bounds = zono_bounds(Z.∂Z)
    # ∂bounds₁₂ = zeros(size(Z.Z₂.G,1))
    # for i in 1:input_dim
    #     ∂bounds₁₂ .+= abs.(Z.Z₁.G[:,i].-Z.Z₂.G[:,i])
    # end
    # for i in 1:Z.num_approx₁
    #     ∂bounds₁₂ .+= abs.(Z.Z₁.G[:,input_dim+i])
    # end
    # for i in 1:Z.num_approx₂
    #     ∂bounds₁₂ .+= abs.(Z.Z₂.G[:,input_dim+i])
    # end
    # ∂bounds[:,1] = max.(∂bounds[:,1],(Z.Z₁.c .- Z.Z₂.c) .- ∂bounds₁₂)
    # ∂bounds[:,2] = min.(∂bounds[:,2],(Z.Z₁.c .- Z.Z₂.c) .+ ∂bounds₁₂)
    lower₁ = @view bounds₁[:,1]
    upper₁ = @view bounds₁[:,2]
    lower₂ = @view bounds₂[:,1]
    upper₂ = @view bounds₂[:,2]
    ∂lower = @view ∂bounds[:,1]
    ∂upper = @view ∂bounds[:,2]

    # Compute Phase Behaviour
    neg₁ = (upper₁ .<= 0.0)
    pos₁ = (lower₁ .>= 0.0)
    any₁ = (!).(neg₁) .&& (!).(pos₁)
    neg₂ = (upper₂ .<= 0.0)
    pos₂ = (lower₂ .>= 0.0)
    any₂ = (!).(neg₂) .&& (!).(pos₂)

    crossing_new_generator = (any₁ .&& (any₂ .|| pos₂)) .|| (pos₁ .&& any₂)

    # Compute Zonotopes for individual networks
    Z₁_new = L1(Z.Z₁,P;bounds = bounds₁)
    Z₂_new = L2(Z.Z₂,P;bounds = bounds₂)

    # Compute new dimensions
    output_dim = size(Z.Z₂,1)
    num_approx₁ = size(Z₁_new.G,2)-input_dim
    num_approx₁_additional = num_approx₁-Z.num_approx₁
    num_approx₂ = size(Z₂_new.G,2)-input_dim
    num_approx₂_additional = num_approx₂-Z.num_approx₂
    ∂num_approx = Z.∂num_approx+count(crossing_new_generator)
    ∂num_approx_additional = ∂num_approx-Z.∂num_approx

    DEBUG_ANY_POS = false
    DEBUG_POS_ANY = false
    DEBUG_ANY_ANY = false
    

    Ĝ = zeros(Float64,
        output_dim,
        input_dim+num_approx₁+num_approx₂+∂num_approx)
    ĉ = zeros(output_dim)
    
    selector = zeros(Bool,output_dim)


    #influence_new = zeros(Float64, size(Z.∂Z.influence,1), size(Z.∂Z.influence,2)+num_approx₁_additional+num_approx₂_additional+∂num_approx_additional)

    #influence_new[:,1:input_dim] .= Z.∂Z.influence
    #influence_new[:,(input_dim+1):(input_dim+Z.num_approx₁)] .= Z.∂Z.influence[:,(input_dim+1):(input_dim+Z.num_approx₁)]
    #influence_new[:,(input_dim+num_approx₁+1):(input_dim+num_approx₁+Z.num_approx₂)] .= Z.∂Z.influence[:,(input_dim+Z.num_approx₁+1):(input_dim+Z.num_approx₁+Z.num_approx₂)]
    #influence_new[:,(input_dim+num_approx₁+num_approx₂+1):(input_dim+num_approx₁+num_approx₂+Z.∂num_approx)] .= Z.∂Z.influence[:,(input_dim+Z.num_approx₁+Z.num_approx₂+1):end]

    # Neg Neg:
    # (Done through default initialization of Ĝ and ĉ)
    # selector .= neg₁ .& neg₂
    # Ĝ[selector] .= 0.0
    # ĉ[selector] .= 0.0

    check = (neg₁ .& neg₂)

    # Neg Pos:
    selector .= neg₁ .& pos₂
    if any(selector)
        # println("NEG_POS")
        Ĝ[selector,1:input_dim] .-= (@view Z₂_new.G[selector,1:input_dim])
        Ĝ[selector,input_dim+num_approx₁+1:input_dim+num_approx₁+num_approx₂] .-= (@view Z₂_new.G[selector,input_dim+1:end])
        ĉ[selector] .-= (@view Z₂_new.c[selector])
        check .|= selector
    end

    # Pos Neg:
    selector .= pos₁ .& neg₂
    if any(selector)
        # println("POS_NEG")
        Ĝ[selector,1:input_dim+num_approx₁] .+= (@view Z₁_new.G[selector,1:end])
        ĉ[selector] .+= (@view Z₁_new.c[selector])
        check .|= selector
    end

    # Pos Pos:
    # This just copies the row from the input ∂Z
    # We also need this for Any Pos and Pos Any and thus we copy for those as well
    selector .= pos₁ .&& (pos₂ .|| any₂) .|| (any₁ .&& pos₂)
    if any(selector)
        # println("POS_POS")
        Ĝ[selector,1:input_dim+Z.num_approx₁] .= (@view Z.∂Z.G[selector,1:input_dim+Z.num_approx₁])
        if Z.num_approx₂ > 0
            Ĝ[selector,input_dim+num_approx₁+1:input_dim+num_approx₁+Z.num_approx₂] .= (@view Z.∂Z.G[selector,input_dim+Z.num_approx₁+1:input_dim+Z.num_approx₁+Z.num_approx₂])
        end
        if Z.∂num_approx > 0
            Ĝ[selector,input_dim+num_approx₁+num_approx₂+1:input_dim+num_approx₁+num_approx₂+Z.∂num_approx] .= (@view Z.∂Z.G[selector,input_dim+Z.num_approx₁+Z.num_approx₂+1:end])
        end
        ĉ[selector] .= (@view Z.∂Z.c[selector])
        check .|= selector
    end

    # Any Neg
    selector .= any₁ .&& neg₂
    if any(selector)
        # println("ANY_NEG")
        Ĝ[selector,1:(input_dim+num_approx₁)] .= (@view Z₁_new.G[selector,1:end])
        ĉ[selector] .= (@view Z₁_new.c[selector])
        check .|= selector
    end

    # Neg Any
    selector .= neg₁ .&& any₂
    if any(selector)
        # println("NEG_ANY")
        Ĝ[selector,1:input_dim] .-= (@view Z₂_new.G[selector,1:input_dim])
        Ĝ[selector,input_dim+num_approx₁+1:input_dim+num_approx₁+num_approx₂] .-= (@view Z₂_new.G[selector,input_dim+1:end])
        ĉ[selector] .-= (@view Z₂_new.c[selector])
        check .|= selector
    end

    instable_new_generators = 0

    generator_offset = input_dim+num_approx₁+num_approx₂+Z.∂num_approx+1
    # Any Pos
    selector .= any₁ .&& pos₂
    instable_new_generators += count(selector)
    if any(selector)
        if DEBUG_ANY_POS
            # println("ANY_POS")
            Ĝ[selector,:] .= 0.0
            ĉ[selector] .= 0.0
            count_generators = count(selector)
            low = max.(0.0,lower₁[selector])-upper₂[selector]
            high = upper₁[selector]-lower₂[selector]
            mid = 0.5 .* (high+low)
            range = 0.5 .* (high-low)
            Ĝ[selector,generator_offset:(generator_offset+count_generators-1)] .= range .* (@view I(output_dim)[selector, selector])
            ĉ[selector] .= mid
            generator_offset += count_generators
        else
            # println("ANY_POS")
            α = -lower₁[selector]
            α ./= (upper₁[selector] .- lower₁[selector])
            Ĝ[selector,1:(input_dim+Z.num_approx₁)] .-= α .* (@view Z.Z₁.G[selector,1:end])
            ĉ[selector] .-= α .* (@view Z.Z₁.c[selector])
            @assert all(α .> 0.0)
            α .*= 0.5 .* upper₁[selector] #max.((-).(lower₁[selector]), upper₁[selector])
            #α .= max.(0.5.*α.*upper₁[selector],(1.0.-α).*((-).(lower₁[selector])))
            count_generators = count(selector)
            Ĝ[selector,(generator_offset:(generator_offset+count_generators-1))] .= (abs.(α)).*(@view I(output_dim)[selector, selector])
            ĉ[selector] .+= α
            generator_offset += count_generators
        end
        check .|= selector
    end

    # Pos Any
    selector .= pos₁ .&& any₂
    instable_new_generators += count(selector)
    if any(selector)
        if DEBUG_POS_ANY
            # println("POS_ANY")
            Ĝ[selector,:] .= 0.0
            ĉ[selector] .= 0.0
            count_generators = count(selector)
            low = lower₁[selector]-upper₂[selector]
            high = upper₁[selector]-max.(0.0,lower₂[selector])
            mid = 0.5 .* (high+low)
            range = 0.5 .* (high-low)
            Ĝ[selector,generator_offset:(generator_offset+count_generators-1)] .= range .* (@view I(output_dim)[selector, selector])
            ĉ[selector] .= mid
            generator_offset += count_generators
        else
            # println("POS_ANY")
            α = -lower₂[selector]
            α ./= (upper₂[selector] .- lower₂[selector])
            Ĝ[selector,1:(input_dim)] .+= α .* (@view Z.Z₂.G[selector,1:input_dim])
            Ĝ[selector,(input_dim+num_approx₁+1):(input_dim+num_approx₁+Z.num_approx₂)] .+= α .* (@view Z.Z₂.G[selector,(input_dim+1):end])
            ĉ[selector] .+= α .* (@view Z.Z₂.c[selector])
            @assert all(α .> 0.0)
            #α .= max.(0.5.*α.*upper₂[selector],(1.0.-α).*((-).(lower₂[selector])))
            α .*= 0.5 .* upper₂[selector] #max.((-).(lower₂[selector]), upper₂[selector])
            #0.5 .* max.((-).(lower₂[selector]), upper₂[selector])
            count_generators = count(selector)
            Ĝ[selector,(generator_offset:(generator_offset+count_generators-1))] .= (abs.(α)).*(@view I(output_dim)[selector, selector])
            ĉ[selector] .-= α
            generator_offset += count_generators
        end
        check .|= selector
    end

    # Any Any
    selector .= any₁ .&& any₂
    instable_new_generators += count(selector)
    if any(selector)
        if DEBUG_ANY_ANY
            # println("ANY_ANY")
            Ĝ[selector,:] .= 0.0
            ĉ[selector] .= 0.0
            count_generators = count(selector)
            low = max.(0.0,lower₁[selector])-upper₂[selector]
            high = upper₁[selector]-max.(0.0,lower₂[selector])
            mid = 0.5 .* (high+low)
            range = 0.5 .* (high-low)
            Ĝ[selector,generator_offset:(generator_offset+count_generators-1)] .= range .* (@view I(output_dim)[selector, selector])
            ĉ[selector] .= mid
        else
            α = ∂upper[selector]
            α ./= (α .- ∂lower[selector])
            # TODO: what's this?
            α .= clamp.(α,0.0,1.0)
            @assert all(α .>= 0.0) && all(α .<= 1.0)
            Ĝ[selector,1:(input_dim+Z.num_approx₁)] .= α .* (@view Z.∂Z.G[selector,1:(input_dim+Z.num_approx₁)])
            if Z.num_approx₂ > 0
                Ĝ[selector,input_dim+num_approx₁+1:input_dim+num_approx₁+Z.num_approx₂] .= α .* (@view Z.∂Z.G[selector,input_dim+Z.num_approx₁+1:input_dim+Z.num_approx₁+Z.num_approx₂])
            end
            if Z.∂num_approx > 0
                Ĝ[selector,input_dim+num_approx₁+num_approx₂+1:input_dim+num_approx₁+num_approx₂+Z.∂num_approx] .= α .* (@view Z.∂Z.G[selector,input_dim+Z.num_approx₁+Z.num_approx₂+1:end])
            end
            ĉ[selector] .= α .* (@view Z.∂Z.c[selector])
            α .*= -min.(0.0,∂lower[selector])
            μ = 0.5 .* max.(∂upper[selector],-∂lower[selector])
            count_generators = count(selector)
            # print(generator_offset)
            # print(count_generators)
            # print(size(Ĝ))
            # print(size(Ĝ[selector,(generator_offset:end)]))
            Ĝ[selector,(generator_offset:end)] .= (abs.(μ)).*(@view I(output_dim)[selector, selector])
            ĉ[selector] .+= α 
            ĉ[selector] .-= μ
        end
        check .|= selector
    end
    if !all(check)
        println("Missed rows:")
        println("Bounds1: ",bounds₁[(!).(check),:])
        println("Bounds2: ",bounds₂[(!).(check),:])
        println("∂Bounds: ",∂bounds[(!).(check),:])
        @assert false
    end

    if FIRST_ROUND
        print("Instable Generators: ",instable_new_generators,"\n")
    end

    ∂Z_new = Zonotope(Ĝ, ĉ, Z.∂Z.influence)
    return DiffZonotope(Z₁_new,Z₂_new, ∂Z_new,num_approx₁,num_approx₂,∂num_approx)

    #α = ∂upper ./ (∂upper - ∂lower)
    # λ = ifelse.((upper₁.<=0.0 .&& upper₂ .<= 0.0) , 0.0, ifelse.(lower₁.>=0.0 .&& lower₂ .>= 0.0, 1.0, ifelse.(∂upper .<= 0.0,0.0,α)))
    
    # Difference between N1 - N2
    # Case 0: Both instable
    # Case 1: N1 zero -> Δ = 0-max(0,x1 - Δᵢ) = 0 or Δᵢ - x1 >= Δᵢ
    #δ = 0.5 .* max.(∂upper,-∂lower) #ifelse.(∂upper>-∂lower, 0.5*∂upper, -0.5*∂lower
    #γ = crossing .* (-δ .- λ .*∂lower )
    #ĉ = λ .* Z.∂Z.c + γ
    # Ĝ = λ .* Z.∂Z.G
    #Ĝ[:,offset_cols:(offset_cols+num_cols)] .= λ .* (@view Z.∂Z.G[:,offset_cols_z:end])
    # New Approx columns
    #Ĝ[:,offset_cols:end] .= (δ.+ sign.(δ)*1e-5) .* (@view I(output_dim)[:, crossing])
    end
end


function (N::GeminiNetwork)(Z :: DiffZonotope, P :: PropState)
    #println("Prop network")
    return foldl((Z,Ls) -> propagate_diff_layer(Ls,Z,P),zip(N.network1.layers,N.diff_network.layers,N.network2.layers),init=Z)
end