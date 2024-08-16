function check_model_feasibility(model, DS, context, file, line)
    optimize!(model)
    error = false
    println("########## DEBUG: $context ##########")
    println("### File: $(file):$(line)")
    println("### Model Check:")
    if termination_status(model) == MOI.INFEASIBLE
        println("### Solver Status: ", termination_status(model))
        println("### ERROR in layer with $(size(Z.Z₁.G,1)) output dimensions")
        error =true
    else
        println("### Solver Status: ", termination_status(model))
        println("### Optimal value: ", objective_value(model))
    end
    if error
        throw("Model check failed!")
    end
end

function check_invariant(Z,DS,context,file,line)
    inv_model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(inv_model, "OutputFlag", 0)
    set_optimizer_attribute(inv_model, "Threads", 1)
    set_optimizer_attribute(inv_model, "LogToConsole", 0)
    @variable(inv_model, -1 <= lp_x[1:size(Z.∂Z.G,2)] <= 1)
    offset = size(Z.Z₁.G,2)+1
    input_dim = size(Z.Z₂,2)-Z.num_approx₂
    Zdiff_result = Z.∂Z.G*lp_x .+ Z.∂Z.c
    Z1_result = (Z.Z₁.G*lp_x[1:size(Z.Z₁.G,2)] .+ Z.Z₁.c)
    Z2_result = if Z.num_approx₂ > 0
                    (
                        Z.Z₂.G[:,1:input_dim]*lp_x[1:input_dim] .+
                        Z.Z₂.G[:,input_dim+1:end]*lp_x[offset:offset+Z.num_approx₂-1] .+
                        Z.Z₂.c
                    )
                else
                    (
                        Z.Z₂.G[:,1:input_dim]*lp_x[1:input_dim] .+
                        Z.Z₂.c
                    )
                end
    Zdiff_result = Zdiff_result[DS.inv_dim_ranges]
    Z1_result = Z1_result[DS.inv_dim_ranges]
    Z2_result = Z2_result[DS.inv_dim_ranges]
    @constraint(inv_model, Zdiff_result .== Z1_result .- Z2_result)
    if !isnothing(DS.propagation_point1)
        @constraint(inv_model, Z1_result .== DS.propagation_point1[DS.inv_dim_ranges])
    end
    if !isnothing(DS.propagation_point2)
        @constraint(inv_model, Z2_result .== DS.propagation_point2[DS.inv_dim_ranges])
    end
    if !isnothing(DS.propagation_point1) && !isnothing(DS.propagation_point2)
        @constraint(inv_model,
            Zdiff_result .== 
            (DS.propagation_point1 .- DS.propagation_point2)[DS.inv_dim_ranges]
        )
    end
    check_model_feasibility(inv_model, DS, context * " (Invariant)", file, line)
end

function print_reludiff_case_decisions(selector,DS,context,file,line)
    if DS.active && !isnothing(DS.print_bound_ranges)
        if any(selector[DS.print_bound_ranges])
            println("########## DEBUG: $context ##########")
            println("### File: $(file):$(line)")
            println("### ReluDiff Case Decisions:")
            println("### Decisions: ", DS.print_bound_ranges[findall(selector[DS.print_bound_ranges])])
        end
    end
end

function print_bounds(Z,DS,context,file,line)
    if DS.active && !isnothing(DS.print_bound_ranges)
        bounds₁ = zono_bounds(Z.Z₁)
        bounds₂ = zono_bounds(Z.Z₂)
        ∂bounds = zono_bounds(Z.∂Z)
        println("########## DEBUG: $context ##########")
        println("### File: $(file):$(line)")
        println("### Bounds of Node $(DS.print_bound_ranges):")
        println("### Lower: ", bounds₁[DS.print_bound_ranges,1])
        println("### Upper: ", bounds₁[DS.print_bound_ranges,2])
        println("### Bounds of Node $(DS.print_bound_ranges):")
        println("### Lower: ", bounds₂[DS.print_bound_ranges,1])
        println("### Upper: ", bounds₂[DS.print_bound_ranges,2])
        println("### Bounds of Node $(DS.print_bound_ranges):")
        println("### Lower: ", ∂bounds[DS.print_bound_ranges,1])
        println("### Upper: ", ∂bounds[DS.print_bound_ranges,2])
    end
end

function propagate_inspection_points(Ls,DS)
    L1, ∂L, L2 = Ls
    if DS.active && !isnothing(DS.propagation_point1)
        DS.propagation_point1 = L1(DS.propagation_point1)
    end
    if DS.active && !isnothing(DS.propagation_point2)
        DS.propagation_point2 = L2(DS.propagation_point2)
    end
end