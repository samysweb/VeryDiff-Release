import Base: size

function size(Z::Zonotope, d::Integer)
    @assert d<=2
    return size(Z.G,d)
end