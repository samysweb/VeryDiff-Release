using Random

function verify_network(
    N1 :: Network,
    N2 :: Network,
    bounds,
    property_check,
    split_heuristic;
    timeout=Inf)
    global FIRST_ROUND = true
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

    # Property
    # property_check = get_epsilon_property(epsilon;focus_dim=focus_dim)
    # split_heuristic = epsilon_split_heuristic
    # property_check = get_top1_property(1.0)
    # split_heuristic = top1_configure_split_heuristic(3) #epsilon_split_heuristic

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
        (1.0,VerificationTask(mid, distance, non_zero_indices, ∂Z_original, nothing))
    )
    end
    @timeit to "Verify" begin
    if single_threaded
        worker_function(common_state, 1, prop_state,N,N1,N2,
        property_check, split_heuristic,num_threads;timeout=timeout)
    else
        worker_list = []
        for threadid in 1:(num_threads)
            push!(worker_list,Threads.@spawn worker_function(common_state, threadid, prop_state,N,N1,N2,property_check, split_heuristic,num_threads;timeout=timeout))
        end
        println("Thread $(Threads.threadid()) is waiting for termination of workers")
        init_time = time_ns()
        while !common_state.should_exit
            sleep(0.05)
            if (time_ns()-init_time)/1e9 > timeout
                println("\n\nTIMEOUT REACHED")
                println("UNKNOWN")
                invoke_termination(common_state)
            end
        end
        for w in worker_list
            wait(w)
        end
    end
    end
    show(to)
    common_state=nothing
    nothing
end

function worker_function(common_state, threadid, prop_state,N,N1,N2,property_check, split_heuristic, num_threads;timeout=Inf)
    try
        thread_result = @timed worker_function_internal(common_state, threadid, prop_state,N,N1,N2,num_threads, property_check, split_heuristic, timeout=timeout)
        println("[Thread $(threadid)] Finished in $(round(thread_result.time;digits=2))s")
        return nothing
    catch e
        println("[Thread $(threadid)] Caught exception: $(e)")
        showerror(stdout, e, catch_backtrace())
    end
end
function worker_function_internal(common_state, threadid, prop_state,N,N1,N2,num_threads, property_check, split_heuristic ;timeout=Inf)
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
    first=true
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
            if first
                println("Zono Bounds:")
                bounds = zono_bounds(Zout.∂Z)
                println(bounds[:,1])
                println(bounds[:,2])
                first=false
            end

            prop_satisfied, cex, heuristics_info, verification_status = property_check(N1, N2, Zin, Zout, verification_task.verification_status)
            global FIRST_ROUND = false
            if !prop_satisfied
                if !isnothing(cex)
                    println("\nFound counterexample: $(cex)")
                    invoke_termination(common_state)
                else
                    splits += 1
                    split_d = split_heuristic(Zin,Zout,heuristics_info,verification_task.distance_indices)
                    Z1, Z2 = split_zono(split_d, verification_task,work_share,verification_status)
                    Zin=nothing
                    push!(task_queue, Z1)
                    push!(task_queue, Z2)
                    generated_zonos+=2
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
            should_terminate = true
        end
        if total_zonos > 1_000 && iszero(total_work)
            println("No resolved zonotopes after depth 1000 -> aborting")
            println("TIMEOUT")
            println("UNKNOWN")
            empty!(task_queue)
            invoke_termination(common_state)
            should_terminate = true
        end
        k+=1
        if k%100 == 0
            println("[Thread $(threadid)] Processed $(total_zonos) zonotopes (Work Done: $(round(100*total_work;digits=5))%; Expected Zonos: $(total_zonos/total_work))")
        end
        if k%50 == 0
            # Shuffle queue
            shuffle!(task_queue.queue)
        end
        #end
    end
    end
    empty!(task_queue)
    invoke_termination(common_state)
    println("[Thread $(threadid)] Total splits: $(splits)")
    print("Processed $(total_zonos) zonotopes (Work Done: $(round(100*total_work;digits=1))%); Generated $(generated_zonos) (Waited $(round(wait_time;digits=2))s; $(loop_time/k)s/loop)\n")
end

function split_zono(d, verification_task :: VerificationTask, work_share, verification_status)
    #return @timeit to "Split_Zono" begin
    distance_d = findfirst(x->x==d,verification_task.distance_indices)
    @assert !isnothing(distance_d)
    low = verification_task.middle[d]-verification_task.distance[distance_d]
    high = verification_task.middle[d]+verification_task.distance[distance_d]
    mid = verification_task.middle[d]
    mid1 = (low+mid)/2
    distance1 = mid1-low
    distance1_vec = deepcopy(verification_task.distance)
    distance1_vec[distance_d] = distance1
    middle1_vec = deepcopy(verification_task.middle)
    middle1_vec[d] = mid1

    Z1 = VerificationTask(middle1_vec, distance1_vec, verification_task.distance_indices, deepcopy(verification_task.∂Z), verification_status)

    mid2 = (mid+high)/2
    distance2 = mid2-mid
    distance2_vec = verification_task.distance
    distance2_vec[distance_d] = distance2
    middle2_vec = verification_task.middle
    middle2_vec[d] = mid2
    Z2 = VerificationTask(middle2_vec, distance2_vec, verification_task.distance_indices, verification_task.∂Z, deepcopy(verification_status))

    return (work_share/2.0,Z1), (work_share/2.0,Z2)
end