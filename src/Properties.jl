function get_epsilon_property(epsilon;focus_dim=nothing)
    return (N1, N2, Zin, Zout) -> begin
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
                return false, (Zin.Z₁.c, sample_distance), nothing
            else
                return false, nothing, (out_bounds, epsilon, focus_dim)
            end
        else
            return true, nothing, nothing
        end
    end
end

function top1_optimization_task(net2_generator,net2_bias,constraint_matrix,constraint_bias, lambda)
    num_cols1 = size(constraint_matrix,2)
    return (
        sum(abs,
            net2_generator[1:num_cols1] .+
            sum((lambda.^2) .* constraint_matrix,dims=1)[1,:]
            ) +
            sum(abs,net2_generator[num_cols1+1:end]) +
            sum((lambda.^2).*constraint_bias) + net2_bias
    )
end

function get_top1_property(sign=1.0)
    return (N1, N2, Zin, Zout) -> begin
        out_bound1 = zono_bounds(Zout.Z₁).*sign
        out_bound2 = zono_bounds(Zout.Z₂).*sign
        out_bound_diff = zono_bounds(Zout.∂Z).*sign

        #PRECISION_THRESHOLD=1e-5
        #@assert all((out_bound1 .- out_bound_diff)[:,1] - out_bound2[:,2] .<= PRECISION_THRESHOLD .&& PRECISION_THRESHOLD .>= out_bound2[:,1] - (out_bound1 .- out_bound_diff)[:,2]) "Bounds error: $(out_bound1) - $(out_bound_diff) ⊈ $(out_bound2)"
        input_dim = size(Zout.Z₂,2)-Zout.num_approx₂
        num_cols1 = (input_dim+Zout.num_approx₁)
        #approx_cols2 = num_cols1+Zout.num_approx₂
        generator_importance = zeros(input_dim)
        cases = 0
        negative_cases = 0

        for max1_ind in 1:size(out_bound1,1)
            if out_bound1[max1_ind,2] >= minimum(out_bound1[setdiff(1:end,max1_ind),1])
                # max1_ind can be maximal
                for max2_ind in 1:size(out_bound2,1)
                    if max1_ind != max2_ind
                        cases += 1
                        # Check if N1 can have maximum max1_ind while N2 has maximum max2_ind
                        #net2_generator = Zout.Z₁.G[max2_ind,:]-Zout.∂Z.G[max2_ind,:]-(Zout.Z₁.G[max1_ind,:]-Zout.∂Z.G[max1_ind,:])
                        net2_generator = -Zout.∂Z.G[max2_ind,:]+Zout.∂Z.G[max1_ind,:]
                        net2_generator[1:num_cols1] .+= Zout.Z₁.G[max2_ind,:] - Zout.Z₁.G[max1_ind,:]
                        net2_bias = Zout.Z₁.c[max2_ind]-Zout.∂Z.c[max2_ind]-(Zout.Z₁.c[max1_ind]-Zout.∂Z.c[max1_ind])
                        constraint_matrix = -Zout.Z₁.G[setdiff(1:end,max1_ind),:]
                        for i in 1:size(constraint_matrix,1)
                            constraint_matrix[i,:] .+= Zout.Z₁.G[max1_ind,:]
                        end
                        constraint_bias = -Zout.Z₁.c[setdiff(1:end,max1_ind)].+Zout.Z₁.c[max1_ind]
                        net2_generator[1:num_cols1] .+= sum(constraint_matrix,dims=1)[1,:]
                        objective_gradient = (dlambda, lambda) -> begin
                            _ , bound = autodiff(
                                ReverseWithPrimal,
                                top1_optimization_task,
                                Active,
                                Const(net2_generator),Const(net2_bias),Const(constraint_matrix),Const(constraint_bias),
                                Duplicated(lambda,dlambda)
                            )
                            return bound, dlambda
                        end
                        scalarobj = ScalarObjective(fg=objective_gradient)
                        optprob = OptimizationProblem(scalarobj;inplace=true)
                        lambda_init = rand(size(constraint_matrix,1))
                        res = solve(optprob, lambda_init, Adam(alpha=0.01),OptimizationOptions(maxiter=100))
                        if res.info.minimum < 0
                            negative_cases+=1
                        else
                            lambda⁺ = solution(res)
                            lambda_component = sum((lambda⁺.^2) .* constraint_matrix[:,1:input_dim],dims=1)
                            #print(size(lambda_component))
                            generator_importance += abs.(net2_generator[1:input_dim] .+lambda_component[1,:])
                        end
                        #lambda = rand(size(constraint_matrix,1))
                        #dlambda = zero(lambda)
                        #bound, dlambda = objective_gradient(dlambda, lambda)
                        #print("dlambda: ", dlambda ," - Bound: ", bound)
                    end
                end
            end
        end

        println("Negative cases: ",negative_cases, " / ", cases)
        if negative_cases < cases
            argmax_N1 = argmax(N1(Zin.Z₁.c))
            argmax_N2 = argmax(N2(Zin.Z₂.c))
            if argmax_N1 != argmax_N2
                return false, (Zin.Z₁.c, (argmax_N1, argmax_N2)), nothing
            else
                return false, nothing, generator_importance
            end
        else
            return true, nothing, nothing
        end
    end
end

function top1_split_heuristic(Zin,Zout,heuristics_info)
    return argmax(heuristics_info)
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