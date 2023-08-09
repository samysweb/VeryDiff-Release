using TimerOutputs
using LinearAlgebra

using VeryDiff
using VNNLib

# This is still missing the optimizations for instable + linear neurons
function run()
    reset_timer!(VeryDiff.to)
    N_original = parse_network(VNNLib.load_network("./test/examples/networks/acas-1-3.onnx"))

    N32 = VeryDiff.truncate_network(Float32,N_original)
    N16 = VeryDiff.truncate_network(Float16,N_original)

    low =  Float64[0.6, -0.5, -0.5, 0.45, -0.5]
    high = Float64[0.679857769, 0.5, 0.5, 0.5, -0.45]

    mid = (high.+low) ./ 2
    distance = mid .- low

    input_dim = length(low)
    Z_original1 = Zonotope(distance .* Matrix(I,input_dim,input_dim),mid)
    Z_original2 = deepcopy(Z_original1)
    ∂Z_original = Zonotope(Matrix(0.0I,input_dim,input_dim),zeros(Float64,input_dim))

    #Zin = DiffZonotope(Z_original1,Z_original2,deepcopy(∂Z_original),0,0,0)
    N = GeminiNetwork(N32,N16)

    # @timeit VeryDiff.to "NetworkProp" Z = N(Zin, PropState(true))

    # for d in 1:5
    #     println("$(d): $((zono_optimize(-1.0,Z.∂Z,d),zono_optimize(1.0,Z.∂Z,d)))")
    # end

    todolist = [(1.0,DiffZonotope(Z_original1,Z_original2,deepcopy(∂Z_original),0,0,0))]
    work_done = 0.0
    epsilon = 0.05 #1.0 #0.05
    #epsilon=0.001
    total_zonos = 1
    println("")
    prop_state = PropState(true)
    while length(todolist)>0
        #println("-----")
        work_share, Zin = pop!(todolist)
        prop_state.i = 1
        @timeit VeryDiff.to "NetworkProp" Z = N(deepcopy(Zin), prop_state)
        prop_state.first = false
        bounds = zono_bounds(Z.∂Z)
        if total_zonos==1
            for d in 1:5
                lower_diff = zono_optimize(-1.0,Z.∂Z,d)
                upper_diff = zono_optimize(1.0,Z.∂Z,d)
                println(bounds[d,1],",",bounds[d,2])
            end
        end

        done = true

        #println(sum(abs,any(abs.(bounds).>epsilon,dims=2)[:,1].*Z.∂Z.G[:,5:end],dims=2))
        if any(abs.(bounds).>epsilon)
            # println(size(Z.∂Z.G))
            # println(count((Z.∂Z.G[:,6:end].!=0.0),dims=2))
            split_d = argmax(sum(abs,any(abs.(bounds).>epsilon,dims=2)[:,1].*diag(Zin.Z₁.G).*Z.∂Z.G[:,1:5],dims=1)[1,:])
            #println(split_d)
            Z1 = deepcopy(Zin.Z₁)
            low = Z1.c[split_d] - Z1.G[split_d,split_d]
            high = Z1.c[split_d] + Z1.G[split_d,split_d]
            mid = (high+low)/2
            mid1 = (low+mid)/2
            distance1 = mid1-low
            Z1.G[split_d,split_d] = distance1
            Z1.c[split_d] = mid1
            #println("Partition 1: $(d) [$(low),$(mid)] -> $(mid1)±$(distance1)")

            Z2 = deepcopy(Zin.Z₁)
            mid2 = (mid+high)/2
            distance2 = mid2-mid
            Z2.G[split_d,split_d] = distance2
            Z2.c[split_d] = mid2
            #println("Partition 2: $(d) [$(mid),$(high)] -> $(mid2)±$(distance2)")
            push!(todolist,(work_share/2.0,DiffZonotope(Z1,deepcopy(Z1),deepcopy(∂Z_original),0,0,0)))
            push!(todolist,(work_share/2.0,DiffZonotope(Z2,deepcopy(Z2),deepcopy(∂Z_original),0,0,0)))
            total_zonos+=2
            done=false
        else
            # println("Done:")
            # println(size(Z.∂Z.G))
            # println(count((Z.∂Z.G[:,6:end].!=0.0),dims=2))
            work_done+=work_share
        #    println(lower_diff," - ",upper_diff)
        end
        
        # 1-4:
        # NeuroDiff: ~2.25 sec
        # VeryDiff: 0.59 sec
        #if done
        #    println("Done this one")
        #end
        if total_zonos%101==0
            print("\rTotal Zonos so far: $(total_zonos) Work done: $(work_done)")
        end

    end
    println("")
    println("Total Zonos: $(total_zonos)")
end

run()
show(VeryDiff.to)