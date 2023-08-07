function verify_network(
    N :: Network,
    Z_in :: Zonotope,
    P :: PropState,
    C :: Union{Nothing,PropConfig})
    Z_out = N(Z_in,P)
    for i in 1:size(Z_out,2)
        min_res = zono_optimize(-1.0,Z_out,i)
        max_res = zono_optimize(1.0,Z_out,i)
        println("Dim ",i,": [",min_res,",",max_res,"]")
    end
    return low > 0, low, nothing
end