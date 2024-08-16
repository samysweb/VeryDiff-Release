module Debugger
    using ..VeryDiff
    using JuMP
    using Gurobi
    DEBUG_STATE = nothing

    include("Debugger_Functions.jl")

    mutable struct DebugState
        active :: Bool
        inspection_point :: Union{Vector{Float64},Nothing}
        propagation_point1 :: Union{Vector{Float64},Nothing}
        propagation_point2 :: Union{Vector{Float64},Nothing}
        print_bound_ranges :: Union{Int,Vector{Int},Nothing}
        inv_dim_ranges :: Any
        function DebugState()
            new(false, nothing, nothing, nothing, nothing, :)
        end
    end
    function __init__()
        init_debugger()
    end

    function is_debug_active()
        if !("JULIA_DEBUG" in keys(ENV))
            return false
        end
        if typeof(ENV["JULIA_DEBUG"]) == String
            if "VeryDiff" in split(ENV["JULIA_DEBUG"],",")
                return true
            end
        elseif VeryDiff in ENV["JULIA_DEBUG"]
            return true
        elseif "VeryDiff" in ENV["JULIA_DEBUG"]
            return true
        end
        return false
    end

    function init_debugger()
        VeryDiff.Debugger.DEBUG_STATE = DebugState()
        if is_debug_active()
            VeryDiff.Debugger.DEBUG_STATE.active = true
        end
    end

    function _start_debugger()
        global DEBUG_STATE.active = true
    end

    function print_diff_bound_range(range=nothing)
        if typeof(range) == Int
            global DEBUG_STATE.print_bound_ranges = [range]
        else
            global DEBUG_STATE.print_bound_ranges = range
        end
    end

    function set_inspection_point(point)
        global DEBUG_STATE.inspection_point = point
    end

    function set_inv_dim_ranges(ranges)
        global DEBUG_STATE.inv_dim_ranges = ranges
    end

    macro propagation_init_hook(N, prop_state)
        if is_debug_active()
            return esc(:(VeryDiff.Debugger.propagation_init_fn($N, $prop_state)))
        else
            return esc(:(nothing))
        end
    end

    function propagation_init_fn(N, prop_state)
        global DEBUG_STATE
        DS = VeryDiff.Debugger.DEBUG_STATE
        if DS.active && !isnothing(DS.inspection_point)
            DS.propagation_point1 = copy(DS.inspection_point)
            DS.propagation_point2 = copy(DS.inspection_point)
        end
    end

    macro pre_diffzono_prop_hook(considered_zono,context="")
        if is_debug_active()
            return esc(:(VeryDiff.Debugger.pre_diffzono_prop_hook_fn($considered_zono,context=$context,file=$("$(__source__.file)"),line=$("$(__source__.line)"))))
        else
            return esc(:(nothing))
        end
    end

    function pre_diffzono_prop_hook_fn(Z;context="",file="unknown",line="unknown")
        global DEBUG_STATE
        DS = VeryDiff.Debugger.DEBUG_STATE
        print_bounds(Z,DS,context,file,line)
        check_invariant(Z,DS,context,file,line)
    end

    macro post_diffzono_prop_hook(considered_zono,context="")
        if is_debug_active()
            return esc(:(VeryDiff.Debugger.post_diffzono_prop_hook_fn($considered_zono,context=$context,file=$("$(__source__.file)"),line=$("$(__source__.line)"))))
        else
            return esc(:(nothing))
        end
    end
    function post_diffzono_prop_hook_fn(Z;context="",file="unknown",line="unknown")
        global DEBUG_STATE
        DS = VeryDiff.Debugger.DEBUG_STATE
        check_invariant(Z,DS,context,file,line)
    end

    macro diff_layer_inspection_hook(L)
        if is_debug_active()
            return esc(:(VeryDiff.Debugger.diff_layer_inspection_fn($L)))
        else
            return esc(:(nothing))
        end
    end

    function diff_layer_inspection_fn(Ls)
        global DEBUG_STATE
        DS = VeryDiff.Debugger.DEBUG_STATE
        propagate_inspection_points(Ls,DS)
    end

    macro diffrelu_case_hook(selector,context="")
        if is_debug_active()
            return esc(:(VeryDiff.Debugger.diffrelu_case_hook_fn($selector,context=$context,file=$("$(__source__.file)"),line=$("$(__source__.line)"))))
        else
            return esc(:(nothing))
        end
    end

    function diffrelu_case_hook_fn(selector;context="???",file="unknown",line="unknown")
        global DEBUG_STATE
        DS = VeryDiff.Debugger.DEBUG_STATE
        print_reludiff_case_decisions(selector,DS,context,file,line)
    end

    macro inspect_pre_top1_model(model)
        if is_debug_active()
            return esc(:(VeryDiff.Debugger.inspect_pre_top1_model_fn($model,file=$("$(__source__.file)"),line=$("$(__source__.line)"))))
        else
            return esc(:(nothing))
        end
    end

    function inspect_pre_top1_model_fn(model;file="unknown",line="unknown")
        global DEBUG_STATE
        DS = VeryDiff.Debugger.DEBUG_STATE
        check_model_feasibility(model,DS,"Pre Top 1 Model",file,line)
    end

end