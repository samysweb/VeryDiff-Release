#using Random

@enum VerificationStatus UNKNOWN SAFE UNSAFE

function verify_network(
    N1 :: Network,
    N2 :: Network,
    bounds,
    property_check,
    split_heuristic;
    timeout=Inf)
    global FIRST_ROUND = true
    verification_result = nothing
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
    ∂Z_original = Zonotope(Matrix(0.0I,input_dim,size(non_zero_indices,1)),zeros(Float64,input_dim),nothing)
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
    #single_threaded = num_threads == 1
    #if single_threaded
    #    common_state = MultiThreaddedQueue(1)
    #else
    #    common_state = MultiThreaddedQueue(num_threads)
    #end
    work_queue = Queue()
    push!(work_queue,
        (1.0,VerificationTask(mid, distance, non_zero_indices, ∂Z_original, nothing, 1.0))
    )
    end
    @timeit to "Verify" begin
    @Debugger.propagation_init_hook(N, prop_state)
    #if single_threaded
    verification_result = worker_function(work_queue, 1, prop_state,N,N1,N2, property_check, split_heuristic,num_threads;timeout=timeout)
    if verification_result == SAFE
        println("SAFE")
    elseif verification_result == UNSAFE
        println("UNSAFE")
    else
        println("UNKNOWN")
    end
    # else
    #     worker_list = []
    #     for threadid in 1:(num_threads)
    #         push!(worker_list,Threads.@spawn worker_function(common_state, threadid, prop_state,N,N1,N2,property_check, split_heuristic,num_threads;timeout=timeout))
    #     end
    #     println("Thread $(Threads.threadid()) is waiting for termination of workers")
    #     init_time = time_ns()
    #     while !common_state.should_exit
    #         sleep(0.05)
    #         if (time_ns()-init_time)/1e9 > timeout
    #             println("\n\nTIMEOUT REACHED")
    #             println("UNKNOWN")
    #             #invoke_termination(common_state)
    #         end
    #     end
    #     for w in worker_list
    #         wait(w)
    #     end
    # end
    end
    show(to)
    #common_state=nothing
    work_queue=nothing
    return verification_result
end

function worker_function(work_queue, threadid, prop_state,N,N1,N2,property_check, split_heuristic, num_threads;timeout=Inf)
    try
        thread_result = worker_function_internal(work_queue, threadid, prop_state,N,N1,N2,num_threads, property_check, split_heuristic, timeout=timeout)
        return thread_result
    catch e
        println("[Thread $(threadid)] Caught exception: $(e)")
        showerror(stdout, e, catch_backtrace())
        thread_result = UNKNOWN
        return thread_result
    end
end
function worker_function_internal(work_queue, threadid, prop_state,N,N1,N2,num_threads, property_check, split_heuristic ;timeout=Inf)
    # @debug "Worker initiated on thread $(threadid)"
    starttime = time_ns()
    prop_state = deepcopy(prop_state)
    k = 0
    total_zonos=0
    generated_zonos = 0
    splits = 0
    is_verified = SAFE
    # @debug "[Thread $(threadid)] Starting worker"
    #task_queue = Queue()
    # @debug "[Thread $(threadid)] Syncing queues"
    #should_terminate = sync_queues!(threadid, qork, task_queue)
    #sync_res = @timed sync_queues!(threadid, common_state, task_queue)
    should_terminate = length(work_queue) == 0
    do_not_split = false
    #wait_time = sync_res.time
    total_work = 0.0
    first=true
    # @debug "[Thread $(threadid)] Initiating loop"
    @timeit to "Zonotope Loop" begin
    loop_time = @elapsed begin
    while !should_terminate
        input_dim=1
        try
            work_share, verification_task = pop!(work_queue)
            Zin = to_diff_zono(verification_task)
            input_dim = size(Zin.Z₁.G,2)
            if k == 0
                println("[Thread $(threadid)] Time to first task: $(round((time_ns()-starttime)/1e9;digits=2))s")
            end
            # @debug "[Thread $(threadid)] got work share $(work_share) running on $(Threads.threadid())"
            #println("Processing task on thread $(threadid)")
            total_zonos+=1
            # Initial Pass
            #prop_state.i = 1
            @timeit to "Zonotope Propagate" begin
            Zout = N(Zin, prop_state)
            end
            if first
                println("Zono Bounds:")
                bounds = zono_bounds(Zout.∂Z)
                println(bounds[:,1])
                println(bounds[:,2])
                first=false
            end

            @timeit to "Property Check" begin
            prop_satisfied, cex, heuristics_info, verification_status, distance_bound = property_check(N1, N2, Zin, Zout, verification_task.verification_status)
            end
            global FIRST_ROUND = false
            if !prop_satisfied
                if !isnothing(cex)
                    @assert all(zono_bounds(Zin.Z₁)[:,1] .<= cex[1] .&& cex[1] .<= zono_bounds(Zin.Z₂)[:,2])
                    println("\nFound counterexample: $(cex)")
                    should_terminate = true
                    is_verified = UNSAFE
                elseif !do_not_split
                    @timeit to "Compute Split" begin
                    splits += 1
                    split_d = split_heuristic(Zin,Zout,heuristics_info,verification_task.distance_indices)
                    Z1, Z2 = split_zono(split_d, verification_task,work_share,verification_status, distance_bound)
                    Zin=nothing
                    push!(work_queue, Z1)
                    push!(work_queue, Z2)
                    generated_zonos+=2
                    end
                end
            else
                total_work += work_share
            end
        finally
            # nothing right now
        end
        if (time_ns()-starttime)/1e9 > timeout
            println("\n\nTIMEOUT REACHED")
            println("UNKNOWN")
            should_terminate = true
            is_verified = UNKNOWN
        end
        # Even if we haven't done any work we may find a counterexample, but unfortunately
        # memory is bounded...
        if length(work_queue) > 2500 && total_work < 1e-3 && input_dim > 100
            println("2500 Zonotopes in task queue: NO MORE SPLITTING!")
            println("This is to avoid memory overflows.")
            println("WARNING: FROM THIS POINT ONWARDS, WE ARE ONLY SEARCHING FOR COUNTEREXAMPLES!")
            do_not_split = true
        end
        should_terminate |= length(work_queue) == 0
        k+=1
        if k%100 == 0
            println("[Thread $(threadid)] Processed $(total_zonos) zonotopes (Work Done: $(round(100*total_work;digits=5))%; Expected Zonos: $(total_zonos/total_work))")
        end
        #end
    end
    end
    end
    empty!(work_queue)
    if do_not_split
        println("\n\nTIMEOUT REACHED")
        println("UNKNOWN")
        is_verified = UNKNOWN
    end
    println("[Thread $(threadid)] Total splits: $(splits)")
    print("Processed $(total_zonos) zonotopes (Work Done: $(round(100*total_work;digits=1))%); Generated $(generated_zonos) ($(loop_time/k)s/loop)\n")
    return is_verified
end

function split_zono(d, verification_task :: VerificationTask, work_share, verification_status, distance_bound)
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

    Z1 = VerificationTask(middle1_vec, distance1_vec, verification_task.distance_indices, deepcopy(verification_task.∂Z), verification_status, distance_bound)

    mid2 = (mid+high)/2
    distance2 = mid2-mid
    distance2_vec = verification_task.distance
    distance2_vec[distance_d] = distance2
    middle2_vec = verification_task.middle
    middle2_vec[d] = mid2
    Z2 = VerificationTask(middle2_vec, distance2_vec, verification_task.distance_indices, verification_task.∂Z, deepcopy(verification_status), distance_bound)

    return (work_share/2.0,Z1), (work_share/2.0,Z2)
end