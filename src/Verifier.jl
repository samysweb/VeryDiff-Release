function verify_network(
    N1 :: Network,
    N2 :: Network,
    bounds,
    epsilon::Float64;
    timeout=Inf,focus_dim=nothing)
    # Timing
    reset_timer!(to)
    @timeit to "Initialize" begin
    # Prepare Zonotope Initialization
    #@timeit to "Prep_Zono_Init" begin
    low = @view bounds[:,1]
    high = @view bounds[:,2]
    mid = (high.+low) ./ 2
    distance = mid .- low
    input_dim = length(low)
    #end

    # Initialize Zonotope
    #@timeit to "Zono_Init" begin
    non_zero_indices = findall((!).(iszero.(distance)))
    distance = distance[non_zero_indices]
    #Z_original1 = Zonotope(distance .* Matrix(I,input_dim,input_dim)[:,non_zero_distances],mid)
    #Z_original2 = deepcopy(Z_original1)
    ∂Z_original = Zonotope(Matrix(0.0I,input_dim,size(non_zero_indices,1)),zeros(Float64,input_dim))
    #end

    #@timeit to "Network_Init" begin
    N = GeminiNetwork(N1,N2)
    #end

    # Statistics
    #@timeit to "Statistics_Init" begin
    work_done = 0.0
    total_zonos = 1
    #end

    #Config
    prop_state = PropState(true)
    num_threads = Threads.nthreads()
    println("Running with $(num_threads) threads")
    single_threaded = num_threads == 1
    if single_threaded
        common_state = MultiThreaddedQueue(Tuple{Float64,VerificationTask},1)
    else
        common_state = MultiThreaddedQueue(Tuple{Float64,VerificationTask},num_threads)
    end
    push!(common_state.common_queue,
        (1.0,VerificationTask(mid, distance, non_zero_indices, ∂Z_original))
    )
    end
    @timeit to "Verify" begin
    if single_threaded
        worker_function(common_state, 1, prop_state,N,N1,N2,
        epsilon,num_threads;timeout=timeout,focus_dim=focus_dim)
    else
        worker_list = []
        for threadid in 1:(num_threads)
            push!(worker_list,Threads.@spawn worker_function(common_state, threadid, prop_state,N,N1,N2,epsilon,num_threads;timeout=timeout,focus_dim=focus_dim))
        end
        println("Thread $(Threads.threadid()) is waiting for termination of workers")
        init_time = time_ns()
        while !common_state.should_exit
            #GC.safepoint()
            sleep(0.05)
            if (time_ns()-init_time)/1e9 > timeout
                println("\n\nTIMEOUT REACHED")
                println("UNKNOWN")
                invoke_termination(common_state)
            end
            #println("Thread $(Threads.threadid()) is waiting for termination of workers $(worker_list)")
        end
        #worker_function(common_state, prop_state,N,N1,N2, epsilon)
        for w in worker_list
            wait(w)
        end
        #println(worker_list)
    end
    end
    #worker_function(common_state, prop_state,N,N1,N2, epsilon)
    show(to)
    #end
end

function worker_function(common_state, threadid, prop_state,N,N1,N2,epsilon, num_threads;timeout=Inf,focus_dim=nothing)
    try
        thread_result = @timed worker_function_internal(common_state, threadid, prop_state,N,N1,N2,epsilon, num_threads,timeout=timeout,focus_dim=focus_dim)
        println("[Thread $(threadid)] Finished in $(round(thread_result.time;digits=2))s")
        return thread_result.value
    catch e
        println("[Thread $(threadid)] Caught exception: $(e)")
        showerror(stdout, e, catch_backtrace())
    end
end
function worker_function_internal(common_state, threadid, prop_state,N,N1,N2,epsilon, num_threads;timeout=Inf,focus_dim=nothing)
    # @debug "Worker initiated on thread $(threadid)"
    starttime = time_ns()
    prop_state = deepcopy(prop_state)
    k = 0
    total_zonos=0
    generated_zonos = 0
    splits = 0
    # @debug "[Thread $(threadid)] Starting worker"
    task_queue = Queue(Tuple{Float64,VerificationTask})
    # @debug "[Thread $(threadid)] Syncing queues"
    should_terminate = sync_queues!(threadid, common_state, task_queue)
    sync_res = @timed sync_queues!(threadid, common_state, task_queue)
    should_terminate = sync_res.value
    wait_time = sync_res.time
    total_work = 0.0
    # @debug "[Thread $(threadid)] Initiating loop"
    loop_time = @elapsed begin
    while !should_terminate
        try
            work_share, verification_task = pop!(task_queue)
            Zin = to_diff_zono(verification_task)
            if k == 0
                println("[Thread $(threadid)] Time to first task: $(round((time_ns()-starttime)/1e9;digits=2))s")
            end
            # @debug "[Thread $(threadid)] got work share $(work_share) running on $(Threads.threadid())"
            #println("Processing task on thread $(threadid)")
            total_zonos+=1
            # Initial Pass
            #prop_state.i = 1
            #@timeit to "NetworkProp" 
            Zout = N(Zin, prop_state)
            #println("Propagated Zonotope on thread $(threadid)")
            # First round?
            out_bounds = zono_bounds(Zout.∂Z)
            if isone(work_share)
                # Print out initial bounds
                println("[",join([x for x in out_bounds[:,1]],","),"]")
                println("[",join([x for x in out_bounds[:,2]],","),"]")
            end
            #prop_state.first = false
            # Larger than epsilon?
            #@timeit to "PostProp" begin
            distance_bound = if !isnothing(focus_dim)
                    maximum(abs.(out_bounds[focus_dim,:]))
                else
                    maximum(abs.(out_bounds))
                end
            #println("Distance ($focus_dim): $(distance_bound)")
            if distance_bound > epsilon
                #println("Splitting on thread $(threadid)")
                # Is concrete example larger than epsilon?
                sample_distance = if !isnothing(focus_dim)
                    abs.(N1(Zin.Z₁.c)[focus_dim]-N2(Zin.Z₂.c)[focus_dim])
                else
                    maximum(abs.(N1(Zin.Z₁.c)-N2(Zin.Z₂.c)))
                end
                if sample_distance>epsilon
                    # Concrete example is larger than epsilon
                    println("\nFound counterexample: $(Zin.Z₁.c): Distance $(sample_distance)")
                    invoke_termination(common_state)
                else
                    splits += 1
                    split_d = get_splitting(Zin,Zout,out_bounds,epsilon;focus_dim=focus_dim)
                    
                    Z1, Z2 = split_zono(split_d, verification_task,work_share)
                    Zin=nothing
                    push!(task_queue, Z1)
                    push!(task_queue, Z2)
                    generated_zonos+=2
                    #end
                end
            else
                total_work += work_share
            end
        finally
            #if k%100 == 0
            sync_res = @timed sync_queues!(threadid, common_state, task_queue)
            should_terminate = sync_res.value
            wait_time += sync_res.time
            #end
        end
        if (time_ns()-starttime)/1e9 > timeout
            println("\n\nTIMEOUT REACHED")
            println("UNKNOWN")
            invoke_termination(common_state)
        end
        k+=1
        if k%100 == 0
            println("[Thread $(threadid)] Processed $(total_zonos) zonotopes (Work Done: $(round(100*total_work;digits=1))%)")
        end
        #end
    end
    end
    println("[Thread $(threadid)] Total splits: $(splits)")
    print("Processed $(total_zonos) zonotopes (Work Done: $(round(100*total_work;digits=1))%); Generated $(generated_zonos) (Waited $(round(wait_time;digits=2))s; $(loop_time/k)s/loop)\n")
end

function get_splitting(Zin,Zout,out_bounds,epsilon;focus_dim=nothing)
    #return @timeit to "Split_Heuristic"
    input_dim = size(Zin.Z₁.G,2)
    #if isnothing(focus_dim)
        return argmax(
            #max.(
            #abs.(diag(Zin.Z₁.G)).*sum(abs,any(abs.(out_bounds).>epsilon,dims=2)[:,1].*Zout.∂Z.G[:,1:5],dims=1)[1,:],
            #.+
            abs.(sum(Zin.Z₁.G,dims=1)).*sum(abs,any(abs.(out_bounds).>epsilon,dims=2)[:,1].*(Zout.Z₁.G[:,1:input_dim] .- Zout.Z₂.G[:,1:input_dim] ),dims=1)[1,:]
            #)
        )[1]
    # else
    #     return argmax(
    #         #max.(
    #         #abs.(diag(Zin.Z₁.G)).*sum(abs,any(abs.(out_bounds).>epsilon,dims=2)[:,1].*Zout.∂Z.G[:,1:5],dims=1)[1,:],
    #         #.+
    #         abs.(sum(Zin.Z₁.G,dims=1)).*abs.(Zout.Z₁.G[focus_dim,1:input_dim] .- Zout.Z₂.G[focus_dim,1:input_dim] )
    #         #)
    #     )[1]
    # end
end

function split_zono(d, verification_task :: VerificationTask, work_share)
    #return @timeit to "Split_Zono" begin
    distance_d = d # findfirst(x->x==d,verification_task.distance_indices)
    low = verification_task.middle[d]-verification_task.distance[distance_d]
    high = verification_task.middle[d]+verification_task.distance[distance_d]
    mid = verification_task.middle[d]
    mid1 = (low+mid)/2
    distance1 = mid1-low
    distance1_vec = deepcopy(verification_task.distance)
    distance1_vec[distance_d] = distance1
    middle1_vec = deepcopy(verification_task.middle)
    middle1_vec[d] = mid1

    Z1 = VerificationTask(middle1_vec, distance1_vec, verification_task.distance_indices, deepcopy(verification_task.∂Z))

    mid2 = (mid+high)/2
    distance2 = mid2-mid
    distance2_vec = verification_task.distance
    distance2_vec[distance_d] = distance2
    middle2_vec = verification_task.middle
    middle2_vec[d] = mid2
    Z2 = VerificationTask(middle2_vec, distance2_vec, verification_task.distance_indices, verification_task.∂Z)

    return (work_share/2.0,Z1), (work_share/2.0,Z2)
    #end
end