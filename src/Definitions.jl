mutable struct Zonotope
    G::Matrix{Float64}
    c::Vector{Float64}
end

struct VerificationTask
    middle :: Vector{Float64}
    distance :: Vector{Float64}
    distance_indices :: Vector{Int}
    ∂Z::Zonotope
    verification_status
end

# Z₂ = Z₁ - ∂Z

mutable struct DiffZonotope
    Z₁::Zonotope
    Z₂::Zonotope
    ∂Z::Zonotope
    num_approx₁ :: Int
    num_approx₂ :: Int
    ∂num_approx :: Int
end

mutable struct PropState
    first :: Bool
    i :: Int64
    num_relus :: Int64
    relu_config :: Vector{Int64}
    function PropState(first :: Bool)
        return new(first, 0, 0, Int64[])
    end
end

struct PropConfig

end

function cleanup_network(network1)
    valid_layers = []
    for i in 1:length(network1.layers)
        if network1.layers[i] isa Dense
            if all(isone.(diag(network1.layers[i].W))) && all([all(iszero.(diag(network1.layers[i].W, k))) && all(iszero.(diag(network1.layers[i].W, -k))) for k in 1:size(network1.layers[i].W,1)-1])
                continue
            end
        end
        push!(valid_layers, i)
    end
    print(valid_layers)
    @assert length(valid_layers) == length(network2.layers)
    return Network(network1.layers[valid_layers])
end

struct GeminiNetwork
    network1 :: Network
    network2 :: Network
    diff_network :: Network
    function GeminiNetwork(network1 :: Network, network2 :: Network)
        diff_layers = Layer[]
        if length(network1.layers) > length(network2.layers)
            network1 = cleanup_network(network1)
        elseif length(network2.layers) > length(network1.layers)
            network2 = cleanup_network(network2)
        end
        @assert length(network1.layers) == length(network2.layers)
        for (l1, l2) in zip(network1.layers, network2.layers)
            @assert typeof(l1) == typeof(l2)
            if typeof(l1) == Dense
                @assert size(l1.W) == size(l2.W) "Mismatch in weight matrix size: $(size(l1.W)) vs $(size(l2.W))"
                @assert size(l1.b) == size(l2.b)
                push!(diff_layers, Dense(l1.W .- l2.W, l1.b .- l2.b))
            elseif typeof(l1) == ReLU
                push!(diff_layers, ReLU())
            else
                error("Unsupported layer type")
            end
        end
        return new(network1, network2, Network(diff_layers))
    end
end