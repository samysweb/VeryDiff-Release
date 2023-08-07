using VeryDiff
using LinearAlgebra
using Optim
using Random
using TimerOutputs

Random.seed!(1235);

@timeit VeryDiff.to "Fuzzing" begin
for i in 1:10
    println("Network $(i)")
    layers = Layer[]
    input_dim = rand(1:20)
    output_dim = 10
    cur_dim = input_dim
    n = rand(2:10)
    relus = 0
    networks = []
    for i in 1:n
        if i==n
            new_dim = output_dim
        else
            new_dim = rand(50:100)
        end
        W = randn(Float64,(new_dim,cur_dim))
        b = randn(Float64,new_dim)
        cur_dim = new_dim
        relus += new_dim
        push!(layers, Dense(W,b))
        push!(layers, ReLU())
        N = Network(deepcopy(layers))
        push!(networks,(N,new_dim))
    end
    Z_original = Zonotope(Matrix(1.0I,input_dim,input_dim),zeros(Float64,input_dim))
    for (N,output_dim) in networks
        Z = deepcopy(Z_original)
        prop_state = PropState(true)
        @timeit VeryDiff.to "NetworkProp" Z = N(Z, prop_state)
        bounds = Tuple{Float64,Float64}[]
        for d in 1:output_dim
            push!(bounds,(zono_optimize(-1.0,Z,d),zono_optimize(1.0,Z,d)))
        end
        for k in 1:10000
            x = 2*rand(Float64,input_dim).-1.0
            y = N(x)
            for d in 1:output_dim
                if y[d] < bounds[d][1] || y[d] > bounds[d][2]
                    println("Error: $(y[d]) not in $(bounds[d])")
                    println("Input: $(x)")
                    raise("Found error")
                end
            end
        end
    end
end
end

show(VeryDiff.to)