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