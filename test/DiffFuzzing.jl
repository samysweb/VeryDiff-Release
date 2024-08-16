using VeryDiff
using LinearAlgebra
using Random
using TimerOutputs
using JuMP
#using GLPK
using Gurobi

Random.seed!(1234);
VeryDiff.NEW_HEURISTIC = false

GRB_ENV = Gurobi.Env()
GRBsetintparam(GRB_ENV, "OutputFlag", 0)
GRBsetintparam(GRB_ENV, "LogToConsole", 0)
GRBsetintparam(GRB_ENV, "Threads", 0)

# if !VeryDiff.Debugger.DEBUG_STATE.active
#     println("WARNING: IT IS RECOMMENDED TO RUN THIS SCRIPT WITH DEBUGGING ENABLED")
#     println("THIS INCREASES THE CHANCE OF FINDING ERRORS IN THE CODE!")
# end

function get_weighted_random_vector1(values, weights::Vector{Int},size)
    total = sum(weights)
    rand_values = rand(1:total,size)
    decision = cumsum(weights)
    result = map(x -> values[findfirst(decision .>= x)],rand_values)
    return result
end

@timeit VeryDiff.to "Fuzzing" begin
NET_COUNT = 0
while true
    global NET_COUNT
    next_seed = rand(1:9999)
    Random.seed!(next_seed);
    println("[FUZZER] SEED: $(next_seed)")
    NET_COUNT += 1
    println("[FUZZER] NET COUNT: $(NET_COUNT)")
    layers1 = Layer[]
    layers2 = Layer[]
    input_dim = rand(1:50)
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
        net_choice = get_weighted_random_vector1([1,2,3],[1,1,1],1)[1]
        println("NET CHOICE: $(net_choice)")
        if net_choice == 1 # Independent weights
            W2 = randn(Float64,(new_dim,cur_dim))
            b2 = randn(Float64,new_dim)
            zero_one_rows1 = get_weighted_random_vector1([0.0,1.0],[1,9],new_dim)
            simple_rows = get_weighted_random_vector1([0.0,1.0],[4,6],new_dim) .* rand([-1.0,0.0,1.0],(new_dim,cur_dim))
            W1 = W1 .* zero_one_rows1 .+ (1 .- zero_one_rows1) .* simple_rows
            zero_one_rows2 = get_weighted_random_vector1([0.0,1.0],[1,9],new_dim)
            simple_rows = get_weighted_random_vector1([0.0,1.0],[1,9],new_dim) .* rand([-1.0,0.0,1.0],(new_dim,cur_dim))
            W2 = W2 .* zero_one_rows2 .+ (1 .- zero_one_rows2) .* simple_rows
            b1 = b1 .* zero_one_rows1
            b2 = b2 .* zero_one_rows2
        elseif net_choice == 2 # Pruned weights
            W2 = copy(W1)
            b2 = copy(b1)
            zero_one_rows2 = get_weighted_random_vector1([0.0,1.0],[1,9],new_dim)
            simple_rows = get_weighted_random_vector1([0.0,1.0],[1,9],new_dim) .* rand([-1.0,0.0,1.0],(new_dim,cur_dim))
            W2 = W2 .* zero_one_rows2 .+ (1 .- zero_one_rows2) .* simple_rows
            b2 = b2 .* zero_one_rows2
        else
            W2 = W1 + 1e-1*randn(Float64,(new_dim,cur_dim))
            b2 = b1 + 1e-1*randn(Float64,new_dim)
        end


        cur_dim = new_dim
        relus += new_dim
        push!(layers1, Dense(W1,b1))
        push!(layers2, Dense(W2,b2))
        N1 = Network(deepcopy(layers1))
        N2 = Network(deepcopy(layers2))
        push!(networks,(N1,N2,new_dim))
        push!(layers1, ReLU())
        push!(layers2, ReLU())
        N1 = Network(deepcopy(layers1))
        N2 = Network(deepcopy(layers2))
        push!(networks,(N1,N2,new_dim))
    end
    for (N1,N2,output_dim) in networks
        println("----")
        N = GeminiNetwork(N1,N2)
        range = rand([0.1,4])
        offset = rand(input_dim)
        Z_original1 = Zonotope(Matrix(range*I,input_dim,input_dim),offset,nothing)
        Zin = deepcopy(Z_original1)
        Z_original2 = deepcopy(Z_original1)
        ∂Z_original = Zonotope(Matrix(0.0I,input_dim,input_dim),zeros(Float64,input_dim),nothing)
        Z = DiffZonotope(Z_original1,Z_original2,∂Z_original,0,0,0)
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
        threshold=1e-4
        for k in 1:10000
            #x = 2*0.1*rand(Float64,input_dim).-0.1
            x_in = rand(Float64,input_dim)
            x = Zin.G*x_in + Zin.c
            y1 = N1(x)
            y2 = N2(x)
            model = Model(() -> Gurobi.Optimizer(GRB_ENV))
            set_optimizer_attribute(model, "OutputFlag", 0)
            @variable(model, -1 <= lp_x[1:size(Z.∂Z.G,2)] <= 1)
            @constraint(model, lp_x[1:input_dim] .== x_in)
            @constraint(model, Z.∂Z.G*lp_x .+ Z.∂Z.c .== y1.-y2)
            @constraint(model, Z.Z₁.G*lp_x[1:size(Z.Z₁.G,2)] .+ Z.Z₁.c .== y1)
            offset = size(Z.Z₁.G,2)+1
            @constraint(model, 
                Z.Z₂.G[:,1:input_dim]*lp_x[1:input_dim] .+
                Z.Z₂.G[:,input_dim+1:end]*lp_x[offset:offset+Z.num_approx₂-1] .+
                Z.Z₂.c .== y2)
            optimize!(model)

            error = false
            if termination_status(model) == MOI.INFEASIBLE
                println("Solver Status: ", termination_status(model))
                error =true
            end
            # println("Solver Status: ", termination_status(model))

            for d in 1:output_dim
                if y1[d] < bounds1[d][1]-threshold || y1[d] > bounds1[d][2]+threshold
                    println("Dimension $(d)")
                    println("Error: $(y1[d]) not in Net 1 bounds $(bounds1[d])")
                    println("Input: $(x)")
                    error = true
                end
                if y2[d] < bounds2[d][1]-threshold || y2[d] > bounds2[d][2]+threshold
                    println("Dimension $(d)")
                    println("Error: $(y2[d]) not in Net 2 bounds $(bounds2[d])")
                    println("Input: $(x)")
                    error = true
                end
                diff_y = y1 - y2
                if diff_y[d] < ∂bounds[d][1]-threshold || diff_y[d] > ∂bounds[d][2]+threshold
                    println("Dimension $(d)")
                    println("Error: $(diff_y[d]) not in Diff bounds $(∂bounds[d])")
                    println("Input: $(x)")
                    error = true
                end
                if error
                    throw("Found error")
                end
                
            end
        end
    end
end
end

show(VeryDiff.to)