function get_epsilon_property(epsilon;focus_dim=nothing)
    return (N1, N2, Zin, Zout, verification_status) -> begin
        #TODO: Use verification status to ignore proven epsilons
        out_bounds = zono_bounds(Zout.∂Z)
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
                return false, (Zin.Z₁.c, sample_distance), nothing, nothing
            else
                return false, nothing, (out_bounds, epsilon, focus_dim), nothing
            end
        else
            return true, nothing, nothing, nothing
        end
    end
end

function get_top1_property()
    return (N1, N2, Zin, Zout, verification_status) -> begin
        if isnothing(verification_status)
            verification_status = Dict{Tuple{Int,Int},Bool}()
        end
        input_dim = size(Zout.Z₂,2)-Zout.num_approx₂
        #generator_importance = zeros(input_dim)
        top_dimension_importance = zeros(size(Zout.Z₁.G,1))
        other_dimension_importance = zeros(size(Zout.Z₁.G,1))
        argmax_N1 = argmax(N1(Zin.Z₁.c))
        argmax_N2 = argmax(N2(Zin.Z₂.c))
        if argmax_N1 != argmax_N2
            print("Found cex")
            return false, (Zin.Z₁.c, (argmax_N1, argmax_N2)), nothing, nothing
        end
        property_satisfied = true
        for top_index in 1:size(Zout.Z₁,1)
            # TODO: Construct LP that ensures that top_index is maximal in N1
            G1 = Zout.Z₁.G .- Zout.Z₁.G[top_index:top_index,:]
            c1 = Zout.Z₁.c[top_index] .- Zout.Z₁.c
            model = Model(GLPK.Optimizer)
            # GLPK:
            # Processed 1973 zonotopes (Work Done: 100.0%); Generated 1972 (Waited 0.0s; 0.011900008664977191s/loop)
            # [Thread 1] Finished in 23.48s
            # model = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
            # set_attribute(model, "OutputFlag", 0)
            # Gurobi: 
            # Processed 1973 zonotopes (Work Done: 100.0%); Generated 1972 (Waited 0.0s; 0.014410304252407502s/loop)
            # [Thread 1] Finished in 28.43s
                #set_attribute(model, "NumericFocus", 2)
            @variable(model,-1.0 <= x[1:size(Zout.∂Z.G,2)] <= 1.0)
            @constraint(model,G1*x[1:size(Zout.Z₁.G,2)] .<= c1)
            @objective(model,Max,0)
            optimize!(model)
            
            if termination_status(model) == MOI.INFEASIBLE
                for other_index in 1:size(Zout.Z₁,1)
                    verification_status[(top_index,other_index)]=true
                end
            else
                for other_index in 1:size(Zout.Z₁,1)
                    if other_index != top_index && !haskey(verification_status, (top_index,other_index))
                        a = Zout.∂Z.G[top_index,:]-Zout.∂Z.G[other_index,:]
                        a[1:size(Zout.Z₁.G,2)] .+= Zout.Z₁.G[other_index,:]-Zout.Z₁.G[top_index,:]
                        @objective(model,Max,a'*x)
                        optimize!(model)
                        threshold = Zout.Z₁.c[top_index]-Zout.Z₁.c[other_index]+Zout.∂Z.c[other_index]-Zout.∂Z.c[top_index]
                        @assert termination_status(model) != MOI.INFEASIBLE
                        if objective_value(model) < threshold
                            verification_status[(top_index,other_index)]=true
                        else
                            input = Zin.Z₁.G*value.(x[1:input_dim])+Zin.Z₁.c
                            argmax_N1 = argmax(N1(input))
                            argmax_N2 = argmax(N2(input))
                            if argmax_N1 != argmax_N2
                                print("Found cex")
                                return false, (input, (argmax_N1, argmax_N2)), nothing, nothing
                            else
                                #generator_importance .= max.(generator_importance, abs.(a[1:input_dim]))
                                top_dimension_importance[top_index] += 1
                                other_dimension_importance[other_index] + 1
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
        return property_satisfied, nothing, (top_dimension_importance,other_dimension_importance), verification_status
    end
end

function top1_configure_split_heuristic(mode)
    dimension_importance_mode = if mode == 0
        (t,o) -> (t.>0)
    elseif mode == 1
        (t,o) -> (t.>0 .|| o.>0)
    elseif mode == 2
        (t,o) -> t
    elseif mode == 3
        (t,o) -> t+o
    end
    return (Zin,Zout,heuristics_info) -> begin
        top_dimension_importance,other_dimension_importance = heuristics_info
        dimension_importance = dimension_importance_mode(top_dimension_importance,other_dimension_importance)
        input_dim = size(Zin.Z₁.G,2)
        return argmax(
            sum(abs,Zin.Z₁.G,dims=1)[1,:].*sum(abs,dimension_importance.*(Zout.Z₁.G[:,1:input_dim] .- Zout.Z₂.G[:,1:input_dim] ),dims=1)[1,:]
        )[1]
    end
end

function epsilon_split_heuristic(Zin,Zout,heuristics_info)
    out_bounds = heuristics_info[1]
    epsilon = heuristics_info[2]
    focus_dim = heuristics_info[3]
    input_dim = size(Zin.Z₁.G,2)
    return argmax(
        sum(abs,Zin.Z₁.G,dims=1)[1,:].*sum(abs,any(abs.(out_bounds).>epsilon,dims=2)[:,1].*(Zout.Z₁.G[:,1:input_dim] .- Zout.Z₂.G[:,1:input_dim] ),dims=1)[1,:]
    )[1]
end