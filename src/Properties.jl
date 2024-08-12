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
                return false, (Zin.Z₁.c, (N1(cex_input),N2(cex_input),sample_distance)), nothing, nothing, distance_bound
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

function get_top1_property(;scale=one(Float64),naive=false)
    return (N1, N2, Zin, Zout, verification_status) -> begin
        if isnothing(verification_status)
            verification_status = Dict{Tuple{Int,Int},Bool}()
        end
        input_dim = size(Zout.Z₂,2)-Zout.num_approx₂
        #generator_importance = zeros(input_dim)
        top_dimension_violation = zeros(input_dim) #size(Zout.Z₁.G,1))
        #other_dimension_importance = zeros(size(Zout.Z₁.G,1))
        argmax_N1 = argmax(N1(Zin.Z₁.c))
        argmax_N2 = argmax(N2(Zin.Z₂.c))
        if argmax_N1 != argmax_N2
            print("Found cex")
            return false, (Zin.Z₁.c, (argmax_N1, argmax_N2)), nothing, nothing
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
        if !isone(scale)
            dist=log(scale)
        else
            dist=0.0
        end

        for top_index in 1:size(Zout.Z₁,1)
            # TODO: Construct LP that ensures that top_index is maximal in N1
            G1 = Zout.Z₁.G .- Zout.Z₁.G[top_index:top_index,:]
            c1 = Zout.Z₁.c[top_index] .- Zout.Z₁.c

            # Second formulation of "top_index is maximal in N1":
            #
            # G2 = deepcopy(Zout.∂Z.G)
            # G2[:,1:input_dim] .+= Zout.Z₂.G[:,1:input_dim]
            # G2[:,z2_approx_start:z2_approx_end] .+= Zout.Z₂.G[:,(input_dim+1):end]
            # G2 .-= G2[top_index:top_index,:]
            # c2 = (Zout.∂Z.c[top_index] .+ Zout.Z₂.c[top_index]) .- (Zout.∂Z.c .+ Zout.Z₂.c)

            
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
            @variable(model,-1.0 <= x[1:size(Zout.∂Z.G,2)] <= 1.0)
            @constraint(model,G1[1:end .!= top_index,:]*x[1:size(Zout.Z₁.G,2)] .<= (c1[1:end .!= top_index] .-dist))
            
            # Additional (but not that helpful) constraints
            #@constraint(model,G2*x .<= c2)
            #@constraint(model, n2_G*x == n2_c)
            
            
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
                #println("FEASIBLE")
                for other_index in 1:size(Zout.Z₁,1)
                    if other_index != top_index && !haskey(verification_status, (top_index,other_index))
                        if naive
                            a = zeros(size(Zout.∂Z.G,2))
                            a[1:size(Zout.Z₂.G,2)] = Zout.Z₂.G[other_index,:]-Zout.Z₂.G[top_index,:]
                        else
                            a = Zout.∂Z.G[top_index,:]-Zout.∂Z.G[other_index,:]
                            a[1:size(Zout.Z₁.G,2)] .+= Zout.Z₁.G[other_index,:]-Zout.Z₁.G[top_index,:]
                        end
                        violation_difference = a
                        #abs.(Zout.Z₁.G[top_index,1:input_dim] .- Zout.Z₂.G[top_index,1:input_dim]) .+ abs.(Zout.Z₁.G[other_index,1:input_dim] .- Zout.Z₂.G[other_index,1:input_dim])
                        @objective(model,Max,a'*x)
                        # print("Calling GLPK (timeout=$(time_limit_sec(model)))")
                        optimize!(model)
                        # print("Returning from GLPK")
                        threshold = if naive
                            Zout.Z₂.c[top_index]-Zout.Z₂.c[other_index]
                        else
                            threshold = Zout.Z₁.c[top_index]-Zout.Z₁.c[other_index]+Zout.∂Z.c[other_index]-Zout.∂Z.c[top_index]
                        end
                        @assert termination_status(model) != MOI.INFEASIBLE
                        if termination_status(model) != MOI.OPTIMAL
                                #top_dimension_importance[top_index] += 1
                                #other_dimension_importance[other_index] += 1
                                top_dimension_violation .+= abs.(violation_difference[1:input_dim])
                                property_satisfied = false
                                if has_values(model)
                                    distance_bound = max(distance_bound, objective_value(model))
                                end
                                continue
                        end
                        if objective_value(model) < threshold
                            verification_status[(top_index,other_index)]=true
                        else
                            distance_bound = max(distance_bound, objective_value(model))
                            input = Zin.Z₁.G*value.(x[1:input_dim])+Zin.Z₁.c
                            res1 = N1(input)
                            res2 = N2(input)
                            argmax_N1 = argmax(res1)
                            argmax_N2 = argmax(res2)
                            softmax_N1 = exp.(res1)/sum(exp.(res1))
                            second_most = maximum(softmax_N1[1:end .!= argmax_N1])
                            if argmax_N1 != argmax_N2
                                println("Found cex")
                                #println("N1: $(softmax_N1[argmax_N1]) (vs. $second_most)")
                                #println("N2: $(softmax_N1[argmax_N2])")
                                if isone(scale) || abs(softmax_N1[argmax_N1]/second_most) >= scale
                                    println("N1 Scale: $(softmax_N1[argmax_N1]/second_most) >= $scale")
                                    return false, (input, (argmax_N1, argmax_N2)), nothing, nothing, 0.0
                                else
                                    println("Discared cex due to scale ($(softmax_N1[argmax_N1]/second_most) < $scale)")
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