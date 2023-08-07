mutable struct Zonotope
    G::Matrix{Float64}
    c::Vector{Float64}
end

mutable struct DiffZonotope
    Z₁::Zonotope
    Z₂::Zonotope
    ∂Z::Zonotope
    num_approx₁ :: Int
    num_approx₂ :: Int
    ∂num_approx :: Int
end

mutable struct PropState
    empty :: Bool
end

struct PropConfig

end

struct GeminiNetwork
    network1 :: Network
    network2 :: Network
    diff_network :: Network
    function GeminiNetwork(network1 :: Network, network2 :: Network)
        diff_layers = Layer[]
        @assert length(network1.layers) == length(network2.layers)
        for (l1, l2) in zip(network1.layers, network2.layers)
            @assert typeof(l1) == typeof(l2)
            if typeof(l1) == Dense
                @assert size(l1.W) == size(l2.W)
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