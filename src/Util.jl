import Base: size

function size(Z::Zonotope, d::Integer)
    @assert d<=2
    return size(Z.G,d)
end

function truncate_network(T::Type,N::Network)
    layers = []
    for l in N.layers
        if l isa ReLU
            push!(layers, l)
        elseif l isa Dense
            push!(layers,Dense(convert.(Float64,convert.(T,deepcopy(l.W))),convert.(Float64,convert.(T,deepcopy(l.b)))))
        else
            throw("Unknown layer type")
        end
    end
    return Network(layers)
end

function to_diff_zono(task :: VerificationTask)
    input_dim = size(task.middle,1)
    Z1 = Zonotope(Matrix(I, input_dim, input_dim)[:,task.distance_indices] .* task.distance', task.middle, Matrix(1.0I, size(task.distance_indices,1), size(task.distance_indices,1)))
    Z2 = deepcopy(Z1)
    return DiffZonotope(Z1, Z2, task.∂Z, 0, 0, 0)
end