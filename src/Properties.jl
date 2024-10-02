function get_sample_distance(N1, N2, vector, focus_dim=nothing)
    if !isnothing(focus_dim)
        abs.(N1(vector)[focus_dim]-N2(vector)[focus_dim])
    else
        maximum(abs.(N1(vector)-N2(vector)))
    end
end

function get_epsilon_property(epsilon;focus_dim=nothing)
    return (N1, N2, Zin, Zout, verification_status) -> begin
        #TODO: Use verification status to ignore proven epsilons
        out_bounds = zono_bounds(Zout.∂Z)
        input_dim = size(Zin.Z₁.G,2)
        distance_bound, max_dim = if !isnothing(focus_dim)
            maximum(abs.(out_bounds[focus_dim,:])),focus_dim
        else
            maximum(abs.(out_bounds)),argmax(abs.(out_bounds))[1]
        end
        #println("Distance Bound: $distance_bound")
        if distance_bound > epsilon
            cex_input = Zin.Z₁.c
            sample_distance = get_sample_distance(N1, N2, cex_input, focus_dim)
            # for i in 1:size(Zout.Z₁.G,1)
            max_vec = zono_get_max_vector(Zout.Z₁,max_dim)[1:input_dim]
            for c in [-1.0,1.0]
                cex_input = Zin.Z₁.G*(max_vec*c)+Zin.Z₁.c
                sample_distance = max(
                    sample_distance,
                    get_sample_distance(N1, N2, cex_input, focus_dim)
                )
                if sample_distance>epsilon
                    break
                end
            end
            # end
            if sample_distance>epsilon
                return false, (cex_input, (N1(cex_input),N2(cex_input),sample_distance)), nothing, nothing, distance_bound
            end


            return false, nothing, (out_bounds, epsilon, focus_dim), nothing, distance_bound
        else
            return true, nothing, nothing, nothing, distance_bound
        end
    end
end

function get_epsilon_property_naive(epsilon;focus_dim=nothing)
    return (N1, N2, Zin, Zout, verification_status) -> begin
        #TODO: Use verification status to ignore proven epsilons
        b = sum(abs,Zout.Z₁.G[:,1:size(Zout.Z₁.G,2)-Zout.num_approx₁] .- Zout.Z₂.G[:,1:size(Zout.Z₂.G,2)-Zout.num_approx₂];dims=2)
        b += sum(abs,Zout.Z₁.G[:,size(Zout.Z₁.G,2)-Zout.num_approx₁+1:end],dims=2)
        b += sum(abs,Zout.Z₂.G[:,size(Zout.Z₂.G,2)-Zout.num_approx₂+1:end],dims=2)
        out_bounds = [(Zout.Z₁.c .- Zout.Z₂.c)-b (Zout.Z₁.c .- Zout.Z₂.c)+b]
        distance_bound = if !isnothing(focus_dim)
            maximum(abs.(out_bounds[focus_dim,:]))
        else
            maximum(abs.(out_bounds))
        end
        if distance_bound > epsilon
            sample_distance = if !isnothing(focus_dim)
                abs.(N1(Zin.Z₁.c)[focus_dim]-N2(Zin.Z₂.c)[focus_dim])
            else
                maximum(abs.(N1(Zin.Z₁.c)-N2(Zin.Z₂.c)))
            end
            if sample_distance>epsilon
                return false, (Zin.Z₁.c, (N1(Zin.Z₁.c),N2(Zin.Z₂.c),sample_distance)), nothing, nothing, distance_bound
            else
                return false, nothing, (out_bounds, epsilon, focus_dim), nothing, distance_bound
            end
        else
            return true, nothing, nothing, nothing, distance_bound
        end
    end
end

TOP1_FOUND_CONCRETE_DELTA = false

function get_top1_property(;delta=zero(Float64),naive=false)
    if !iszero(delta)
        @assert 0.5 <= delta && delta <= 1.0
        dist=log(delta/(1-delta))
    else
        dist=0.0
    end
    return (N1, N2, Zin, Zout, verification_status) -> begin
        global FIRST_ROUND
        global TOP1_FOUND_CONCRETE_DELTA
        if FIRST_ROUND
            TOP1_FOUND_CONCRETE_DELTA = false
        end
        if isnothing(verification_status)
            verification_status = Dict{Tuple{Int,Int},Bool}()
        end
        input_dim = size(Zout.Z₂,2)-Zout.num_approx₂
        #generator_importance = zeros(input_dim)
        top_dimension_violation = zeros(input_dim) #size(Zout.Z₁.G,1))
        #other_dimension_importance = zeros(size(Zout.Z₁.G,1))
        res1 = N1(Zin.Z₁.c)
        argmax_N1 = argmax(res1)
        argmax_N2 = argmax(N2(Zin.Z₂.c))
        softmax_N1 = exp.(res1)/sum(exp.(res1))
        if argmax_N1 != argmax_N2
            if iszero(delta) || softmax_N1[argmax_N1] >= delta
                println("Found cex")
                println("N1 Probability: $(softmax_N1[argmax_N1]) >= $delta")
                return false, (Zin.Z₁.c, (argmax_N1, argmax_N2)), nothing, nothing, 0.0
            else
                second_largest = sort(res1,rev=true)[2]
                if !iszero(delta) && res1[argmax_N1]-second_largest >= dist
                    println("Found spurious cex")
                    println("N1 Probability: $(softmax_N1[argmax_N1]) < $delta")
                    println("but difference $(res1[argmax_N1]-second_largest) >= $dist (approximate bound)")
                    # println("INCONCLUSIVE")
                    # return false, (Zin.Z₁.c, (argmax_N1, argmax_N2)), nothing, nothing, 0.0
                end
            end
        end
        property_satisfied = true
        distance_bound = 0.0
        # Formulation of "Description of Z₁-∂Z = Z₂" in LP:
        #
        # n2_G = -Zout.∂Z.G
        # n2_G[:,1:size(Zout.Z₁.G,2)] .+= Zout.Z₁.G
        # n2_G[:,1:input_dim] .-= Zout.Z₂.G[:,1:input_dim]
        # z2_approx_start = (size(Zout.Z₁.G,2)+1)
        # z2_approx_end = z2_approx_start + Zout.num_approx₂ - 1
        # n2_G[:,z2_approx_start:z2_approx_end] .-= Zout.Z₂.G[:,(input_dim+1):end]
        # n2_c = -(Zout.Z₁.c .- Zout.∂Z.c) + Zout.Z₂.c

        # δ-Top-1 property
        #out_bounds = zono_bounds(Zout.Z₁)
        #dist=log(0.1)+log(sum(exp.(out_bounds[:,1])))
        #print(out_bounds[:,1])
        #print("Distance: ",dist)
        any_feasible = false
        for top_index in 1:size(Zout.Z₁,1)
            # TODO: Construct LP that ensures that top_index is maximal in N1
            G1 = Zout.Z₁.G .- Zout.Z₁.G[top_index:top_index,:]
            c1 = Zout.Z₁.c[top_index] .- Zout.Z₁.c

            
            #model = Model(GLPK.Optimizer)
            # GLPK:
            # Processed 1973 zonotopes (Work Done: 100.0%); Generated 1972 (Waited 0.0s; 0.011900008664977191s/loop)
            # [Thread 1] Finished in 23.48s
            # set_attribute(GRB_ENV[], "LogToConsole", 0)
            # set_attribute(GRB_ENV[], "OutputFlag", 0)
            # set_attribute(GRB_ENV[], "Method", 1)
            if USE_GUROBI
                model = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
            else
                model = Model(GLPK.Optimizer)
            end
            # Gurobi: 
            # Processed 1973 zonotopes (Work Done: 100.0%); Generated 1972 (Waited 0.0s; 0.014410304252407502s/loop)
            # [Thread 1] Finished in 28.43s
                #set_attribute(model, "NumericFocus", 2)
            set_time_limit_sec(model, 10)
            var_num = size(Zin.Z₁.G,2) + Zout.num_approx₁ + Zout.num_approx₂ + Zout.∂num_approx
            @variable(model,-1.0 <= x[1:var_num] <= 1.0)
            
            # Additional (but not that helpful) constraints
            #@constraint(model,G2*x .<= c2)
            #@constraint(model, n2_G*x == n2_c)
            # Constraint Formulation for:
            # Z₁ = Z₂ + ∂Z <-> G₁*x+c₁ = (G₂+∂G)*x+(c₂+∂c)
            # <-> c₁ - c₂ - ∂c = (G₂+∂G-G₁)*x

            if !naive

                G2 = copy(Zout.∂Z.G)
                offset = 1
                G2[:,offset:input_dim] .+= Zout.Z₂.G[:,1:input_dim] .- Zout.Z₁.G[:,1:input_dim]
                offset += input_dim
                G2[:,offset:(offset+Zout.num_approx₁-1)] .-= Zout.Z₁.G[:,(input_dim+1):end]
                offset += Zout.num_approx₁
                G2[:,offset:(offset+Zout.num_approx₂-1)] .+= Zout.Z₂.G[:,(input_dim+1):end]
                @constraint(model,
                    G2*x .== (Zout.Z₁.c .- Zout.∂Z.c .- Zout.Z₂.c)
                )
            end

            # input_vector = rand([-1.0,1.0],input_dim)

            # G_current = Zout.Z₁.G[[5,8],(input_dim+1):end]
            # c_update = Zout.Z₁.G[[5,8],1:input_dim]*input_vector
            # zono1 = LazySets.Zonotope(Zout.Z₁.c[[5,8]]+c_update,G_current)

            # G_current = Zout.Z₂.G[[5,8],(input_dim+1):end]
            # c_update = Zout.Z₂.G[[5,8],1:input_dim]*input_vector
            # zono2 = LazySets.Zonotope(Zout.Z₂.c[[5,8]]+c_update,G_current)

            # G_zono3 = - Zout.∂Z.G[[5,8],:]
            # G_zono3[:,1:size(Zout.Z₁.G,2)] .+= Zout.Z₁.G[[5,8],:]
            # G_current = G_zono3[:,(input_dim+1):end]
            # c_update = G_zono3[:,1:input_dim]*input_vector
            # zono3 = LazySets.Zonotope(Zout.Z₁.c[[5,8]] .- Zout.∂Z.c[[5,8]]+c_update,G_current)

            # G_current = Zout.∂Z.G[[5,8],(input_dim+1):end]
            # c_update = Zout.∂Z.G[[5,8],1:input_dim]*input_vector
            # ∂zono = LazySets.Zonotope(Zout.∂Z.c[[5,8]]+c_update,G_current)

            # G_zono∂alt = zeros(2,size(Zout.Z₁.G,2)+Zout.num_approx₂)
            # G_zono∂alt[:,1:size(Zout.Z₁.G,2)] .= Zout.Z₁.G[[5,8],:]
            # G_zono∂alt[:,1:input_dim] .-= Zout.Z₂.G[[5,8],1:input_dim]
            # G_zono∂alt[:,(size(Zout.Z₁.G,2)+1):end] .-= Zout.Z₂.G[[5,8],(input_dim+1):end]
            # G_current = G_zono∂alt[:,(input_dim+1):end]
            # c_update = G_zono∂alt[:,1:input_dim]*input_vector
            # zono∂alt = LazySets.Zonotope(Zout.Z₁.c[[5,8]]-Zout.Z₂.c[[5,8]]+c_update,G_current)
            # plt = plot(zono1, label="Z₁",alpha=0.3)
            # plot!(zono2, label="Z₂",alpha=0.3)
            # plot!(zono3, label="Z₂'",alpha=0.3)
            # plot!(∂zono, label="∂Z",alpha=0.3)
            # plot!(zono∂alt, label="∂Z'",alpha=0.3)
            # gui(plt)
            
            Debugger.@inspect_pre_top1_model model
            
            @constraint(model,G1[1:end .!= top_index,:]*x[1:size(Zout.Z₁.G,2)] .<= (c1[1:end .!= top_index] .-dist))
            @objective(model,Max,0)
            
            #@objective(model, Max, sum(Zout.Z₁.G*x[1:size(Zout.Z₁.G,2)]+Zout.Z₁.c,dim=2))
            
            #print("Calling GLPK (timeout=$(time_limit_sec(model)))")
            optimize!(model)
            #print("Returning from GLPK")
            
            if termination_status(model) == MOI.INFEASIBLE
                #println("$top_index INFEASIBLE")
                for other_index in 1:size(Zout.Z₁,1)
                    verification_status[(top_index,other_index)]=true
                end
            else
                
                if !iszero(delta) && !TOP1_FOUND_CONCRETE_DELTA
                    input = Zin.Z₁.G*value.(x[1:input_dim])+Zin.Z₁.c
                    res1 = N1(input)
                    argmax_N1 = argmax(res1)
                    softmax_N1 = exp.(res1)/sum(exp.(res1))
                    if softmax_N1[argmax_N1] >= delta
                        println("[TOP-1] required confidence ($(softmax_N1[argmax_N1])≥$delta) is feasible for index $argmax_N1")
                        TOP1_FOUND_CONCRETE_DELTA=true
                    else
                        println("[TOP-1] did not find required confidence yet.")
                    end
                end
                any_feasible = true
                #println("$top_index FEASIBLE")
                #println("FEASIBLE")
                for other_index in 1:size(Zout.Z₁,1)
                    if other_index != top_index && !haskey(verification_status, (top_index,other_index))
                        # a1 = zeros(var_num)
                        # input_dim = size(Zin.Z₂.G,2)
                        # #print(input_dim)
                        # a1[1:input_dim] .= Zout.Z₂.G[other_index,1:input_dim].-Zout.Z₂.G[top_index,1:input_dim]
                        # offset = input_dim + Zout.num_approx₁ + 1
                        # a1[offset:(offset + Zout.num_approx₂-1)] .= Zout.Z₂.G[other_index,(input_dim+1):end].-Zout.Z₂.G[top_index,(input_dim+1):end]
                        # a2 = Zout.∂Z.G[top_index,:]-Zout.∂Z.G[other_index,:]
                        # a2[1:size(Zout.Z₁.G,2)] .+= Zout.Z₁.G[other_index,:]-Zout.Z₁.G[top_index,:]
                        # plt = plot([a1,a2])
                        # gui(plt)
                        a = zeros(var_num)
                        input_dim = size(Zin.Z₂.G,2)
                        #print(input_dim)
                        a[1:input_dim] .= Zout.Z₂.G[other_index,1:input_dim].-Zout.Z₂.G[top_index,1:input_dim]
                        offset = input_dim + Zout.num_approx₁ + 1
                        a[offset:(offset + Zout.num_approx₂-1)] .= Zout.Z₂.G[other_index,(input_dim+1):end].-Zout.Z₂.G[top_index,(input_dim+1):end]
                        @objective(model,Max,a'*x)
                        violation_difference = a
                        
                        threshold = Zout.Z₂.c[top_index]-Zout.Z₂.c[other_index]
                        # If the optimal value is < threshold, then the property is satisfied
                        # otherwise (optimal >= threshold) we may have found a counterexample
                        if USE_GUROBI # we are using GUROBI -> set objective/bound thresholds
                            set_optimizer_attribute(model, "Cutoff", threshold-1e-6)
                        end
                        optimize!(model)
                        # print("Returning from GLPK")
                        
                        model_status = termination_status(model)
                        # Model must be feasible since we did not add any constraints
                        @assert model_status != MOI.INFEASIBLE
                        # Model should be optimal or have reached the objective limit
                        # any other status -> split and retry
                        if model_status != MOI.OPTIMAL && model_status != MOI.OBJECTIVE_LIMIT
                                println("[GUROBI] Irregular model status: $model_status")
                                #top_dimension_importance[top_index] += 1
                                #other_dimension_importance[other_index] += 1
                                top_dimension_violation .+= abs.(violation_difference[1:input_dim])
                                property_satisfied = false
                                if has_values(model)
                                    distance_bound = max(distance_bound, objective_value(model))
                                end
                                continue
                        end
                        #println("Value: $(objective_value(model))")
                        #println("Threshold: $threshold")
                        if model_status == MOI.OBJECTIVE_LIMIT || objective_value(model) < threshold
                            verification_status[(top_index,other_index)]=true
                        else
                            distance_bound = max(distance_bound, objective_value(model))
                            input = Zin.Z₁.G*value.(x[1:input_dim])+Zin.Z₁.c
                            res1 = N1(input)
                            res2 = N2(input)
                            argmax_N1 = argmax(res1)
                            argmax_N2 = argmax(res2)
                            softmax_N1 = exp.(res1)/sum(exp.(res1))
                            if argmax_N1 != argmax_N2
                                if iszero(delta) || softmax_N1[argmax_N1] >= delta
                                    println("Found cex")
                                    second_most = sort(softmax_N1,rev=true)[2]
                                    println("N1: $(softmax_N1[argmax_N1]) (vs. $second_most)")
                                    softmax_N2 = exp.(res2)/sum(exp.(res2))
                                    println("N2: $(softmax_N2[argmax_N2])")
                                    println("N1 Probability: $(softmax_N1[argmax_N1]) >= $delta")
                                    return false, (input, (argmax_N1, argmax_N2)), nothing, nothing, 0.0
                                else
                                    second_largest = sort(res1,rev=true)[2]
                                    if !iszero(delta) && res1[argmax_N1]-second_largest >= dist
                                        println("Found spurious cex")
                                        println("N1 Probability: $(softmax_N1[argmax_N1]) < $delta")
                                        println("but difference $(res1[argmax_N1]-second_largest) >= $dist (approximate bound)")
                                        # println("INCONCLUSIVE")
                                        # return false, (input, (argmax_N1, argmax_N2)), nothing, nothing, 0.0
                                    end
                                    #println("Discared cex due to probability ($(softmax_N1[argmax_N1]) < $delta)")
                                    #top_dimension_importance[top_index] += 1
                                    #other_dimension_importance[other_index] += 1
                                    top_dimension_violation .+= abs.(violation_difference[1:input_dim])
                                    property_satisfied = false
                                end
                            else
                                #generator_importance .= max.(generator_importance, abs.(a[1:input_dim]))
                                #top_dimension_importance[top_index] += 1
                                #other_dimension_importance[other_index] += 1
                                top_dimension_violation .+= abs.(violation_difference[1:input_dim])
                                property_satisfied = false
                            end
                        end
                    end
                end
            end
            # generator_importance .*= sum(abs,Zin.Z₁.G,dims=1)[1,:]
        end
        @assert !iszero(delta) || any_feasible "One output must be maximal, but our analysis says there is no maximum -- this smells like a bug!"
        # if property_satisfied
        #     println("Zonotope Top 1 Equivalent!")
        # end
        return property_satisfied, nothing, top_dimension_violation, verification_status, distance_bound
    end
end

function top1_configure_split_heuristic(mode)
    # dimension_importance_mode = if mode == 0
    #     (t,o) -> (t.>0)
    # elseif mode == 1
    #     (t,o) -> (t.>0 .|| o.>0)
    # elseif mode == 2
    #     (t,o) -> t
    # elseif mode == 3
    #     (t,o) -> t+o
    # end
    return (Zin,Zout,heuristics_info,distance_indices) -> begin
        top_dimension_violation = heuristics_info
        top_dimension_violation ./= norm(top_dimension_violation,2)
        #dimension_importance = dimension_importance_mode(top_dimension_importance,other_dimension_importance)
        input_dim = size(Zin.Z₁.G,2)
        #∂weights = sum(abs, (Zout.∂Z.G[:,1:input_dim] ),dims=1)[1,:]
        #∂weights ./= norm(∂weights,2)
        #diff_weights = sum(abs, (Zout.Z₁.G[:,1:input_dim] .- Zout.Z₂.G[:,1:input_dim] ),dims=1)[1,:]
        #diff_weights ./= norm(diff_weights,2)

        #diff_weights = sum(abs,Zin.Z₁.G,dims=1)[1,:].*sum(abs,((Zout.Z₁.G)*Zout.Z₁.influence'.-(Zout.Z₂.G)*Zout.Z₂.influence'),dims=1)[1,:]
        if NEW_HEURISTIC
            diff_weights = sum(abs,Zin.Z₁.G,dims=1)[1,:].*sum(abs,(abs.(Zout.Z₁.G)*abs.(Zout.Z₁.influence').+abs.(Zout.Z₂.G)*abs.(Zout.Z₂.influence')),dims=1)[1,:]
            diff_weights ./= norm(diff_weights,2)
        else
            diff_weights = sum(abs, (Zout.Z₁.G[:,1:input_dim] .- Zout.Z₂.G[:,1:input_dim] ),dims=1)[1,:]
            diff_weights ./= norm(diff_weights,2)
        end

        if mode==1
            d = argmax(
                diff_weights
            )[1]
        elseif mode==2
            d = argmax(
                top_dimension_violation
            )[1]
        elseif mode==3
            d = argmax(
                diff_weights .+ top_dimension_violation
            )[1]
        end
        return distance_indices[d]
    end
end

function epsilon_split_heuristic(Zin,Zout,heuristics_info,distance_indices)
    out_bounds = heuristics_info[1]
    epsilon = heuristics_info[2]
    focus_dim = heuristics_info[3]
    input_dim = size(Zin.Z₁.G,2)

    # ∂weights = sum(abs,(Zout.∂Z.G[:,1:input_dim] ),dims=1)[1,:]
    # ∂weights ./= norm(∂weights,2)
    #diff_weights = sum(abs,(Zout.Z₁.G[:,1:input_dim] .- Zout.Z₂.G[:,1:input_dim] ),dims=1)[1,:]
    #diff_weights ./= norm(diff_weights,2)

    #
    #print(size(Zout.Z₁.influence))
    #print(size(Zout.Z₂.influence))
    #println(size(((Zout.Z₁.G[:,:])*Zout.Z₁.influence'.-(Zout.Z₂.G[:,:])*Zout.Z₂.influence')))
    #if isnothing(focus_dim)
    #    relevant_dimensions=any(abs.(out_bounds).>epsilon,dims=2)[:,1]
    #else
    #    relevant_dimensions=focus_dim:(focus_dim)
    #end
    #print(size(relevant_dimensions))
    if NEW_HEURISTIC
        diff_weights = sum(abs,Zin.Z₁.G,dims=1)[1,:].*sum(abs,(abs.(Zout.Z₁.G)*abs.(Zout.Z₁.influence').+abs.(Zout.Z₂.G)*abs.(Zout.Z₂.influence')),dims=1)[1,:]
        # influence1 = sum(abs,(abs.(Zout.∂Z.G[:,1:input_dim])*(abs.(Zout.Z₁.influence'[1:input_dim,:].+Zout.Z₂.influence'[1:input_dim,:]))),dims=1)[1,:]
        # influence2 = sum(abs,(abs.(Zout.∂Z.G[:,(input_dim+1):(input_dim+Zout.num_approx₁)])*(abs.(Zout.Z₁.influence'[(input_dim+1):end,:]))),dims=1)[1,:]
        # influence3 = sum(abs,(abs.(Zout.∂Z.G[:,(input_dim+Zout.num_approx₁+1):(input_dim+Zout.num_approx₁+Zout.num_approx₂)])*(abs.(Zout.Z₂.influence'[(input_dim+1):end,:]))),dims=1)[1,:]
        # diff_weights = sum(abs, Zin.Z₁.G,dims=1)[1,:].*(
        #     influence1 .+
        #     influence2 .+
        #     influence3
        # )
    else
        diff_weights = sum(abs, (Zout.Z₁.G[:,1:input_dim] .- Zout.Z₂.G[:,1:input_dim] ),dims=1)[1,:]
        diff_weights ./= norm(diff_weights,2)
    end


    d = argmax(
        # sum(abs,Zin.Z₁.G,dims=1)[1,:].*
        #(∂weights .+ diff_weights)
        diff_weights
    )[1]

    #print("Selected: $d (vs. $d_alternative)")
    
    return distance_indices[d]
end