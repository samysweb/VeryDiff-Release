using VeryDiff
using LinearAlgebra
using Optim
using Random
using TimerOutputs

Random.seed!(1235);

@timeit VeryDiff.to "Fuzzing" begin
for i in 1:10
    println("Networks $(i)")
    layers1 = Layer[]
    layers2 = Layer[]
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
        W1 = randn(Float64,(new_dim,cur_dim))
        b1 = randn(Float64,new_dim)
        W2 = randn(Float64,(new_dim,cur_dim))
        b2 = randn(Float64,new_dim)
        cur_dim = new_dim
        relus += new_dim
        push!(layers1, Dense(W1,b1))
        push!(layers1, ReLU())
        push!(layers2, Dense(W2,b2))
        push!(layers2, ReLU())
        N1 = Network(deepcopy(layers1))
        N2 = Network(deepcopy(layers2))
        push!(networks,(N1,N2,new_dim))
    end
    Z_original1 = Zonotope(Matrix(1.0I,input_dim,input_dim),zeros(Float64,input_dim))
    Z_original2 = deepcopy(Z_original1)
    ∂Z_original = Zonotope(Matrix(0.0I,input_dim,input_dim),zeros(Float64,input_dim))
    Z_original = DiffZonotope(Z_original1,Z_original2,∂Z_original,0,0,0)
    for (N1,N2,output_dim) in networks
        println("----")
        N = GeminiNetwork(N1,N2)
        Z = deepcopy(Z_original)
        prop_state = PropState(true)
        @timeit VeryDiff.to "NetworkProp" Z = N(Z, prop_state)
        bounds1 = Tuple{Float64,Float64}[]
        bounds2 = Tuple{Float64,Float64}[]
        ∂bounds = Tuple{Float64,Float64}[]
        for d in 1:output_dim
            push!(bounds1,(zono_optimize(-1.0,Z.Z₁,d),zono_optimize(1.0,Z.Z₁,d)))
            push!(bounds2,(zono_optimize(-1.0,Z.Z₂,d),zono_optimize(1.0,Z.Z₂,d)))
            push!(∂bounds,(zono_optimize(-1.0,Z.∂Z,d),zono_optimize(1.0,Z.∂Z,d)))
        end
        for k in 1:10000
            x = 2*rand(Float64,input_dim).-1.0
            y1 = N1(x)
            y2 = N2(x)
            for d in 1:output_dim
                if y1[d] < bounds1[d][1] || y1[d] > bounds1[d][2]
                    println("Dimension $(d)")
                    println("Error: $(y1[d]) not in Net 1 bounds $(bounds[d])")
                    println("Input: $(x)")
                    throw("Found error")
                end
                if y2[d] < bounds2[d][1] || y2[d] > bounds2[d][2]
                    println("Dimension $(d)")
                    println("Error: $(y2[d]) not in Net 2 bounds $(bounds[d])")
                    println("Input: $(x)")
                    throw("Found error")
                end
                diff_y = y1 - y2
                if diff_y[d] < ∂bounds[d][1] || diff_y[d] > ∂bounds[d][2]
                    println("Dimension $(d)")
                    println("Error: $(diff_y[d]) not in Diff bounds $(∂bounds[d])")
                    println("Input: $(x)")
                    throw("Found error")
                end
                
            end
        end
    end
end
end

show(VeryDiff.to)