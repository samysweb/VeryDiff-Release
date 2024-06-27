import Base: push!, length, pop!
import Base.Order.lt
import Base.Order.Ordering
using DataStructures

struct VerificationTaskOrdering <: Ordering end
lt(o::VerificationTaskOrdering, a, b) = b[2].distance_bound < a[2].distance_bound

mutable struct Queue
    queue::BinaryHeap{Tuple{Float64,VerificationTask},VerificationTaskOrdering}
    function Queue()
        return new(BinaryHeap{Tuple{Float64,VerificationTask}}(VerificationTaskOrdering()))
    end
end

function empty!(q::Queue)
    q.queue = BinaryHeap{Tuple{Float64,VerificationTask}}(VerificationTaskOrdering())
end

function push!(q::Queue, x)
    push!(q.queue,x)
end
function pop!(q::Queue)
    return pop!(q.queue)
end
function length(q::Queue)
    return length(q.queue)
end

# mutable struct MultiThreaddedQueue{T}
#     common_queue :: Queue{T}
#     lock :: ReentrantLock
#     condition :: Threads.Condition
#     needs_work :: Vector{Bool}
#     needs_work_any :: Bool
#     should_exit :: Bool
#     function MultiThreaddedQueue(T::Type,n::Int)
#         l = ReentrantLock()
#         return new{T}(Queue(T),l,Threads.Condition(l),ones(Bool,n),true,false)
#     end
# end

# function invoke_termination(state::MultiThreaddedQueue)
#     lock(state.lock) do
#         empty!(state.common_queue)
#         state.should_exit = true
#         notify(state.condition)
#     end
# end

# function sync_queues!(threadid, state::MultiThreaddedQueue, local_queue :: Queue)
#     # @debug "[Thread $(threadid)] Syncing queues"
#     # Case 1: We should just leave
#     if state.should_exit
#         # @debug "[Thread $(threadid)] Case 1: We should just leave"
#         # Return true if we should exit; nothing else to do
#         return true
#     end
#     # Case 2: Work is needed and we can offer it
#     if length(local_queue.queue) > 1 && state.needs_work_any
#         # @debug "[Thread $(threadid)] Case 2: Work is needed and we can offer it"
#         lock(state.lock) do
#             # Drop off necessary number of tasks, but at most length-1
#             task_num = min(count(state.needs_work),length(local_queue.queue)-1)
#             for _ in 1:task_num
#                 push!(state.common_queue.queue,pop!(local_queue.queue))
#             end
#             notify(state.condition)
#         end
#         return false
#     end
#     # Case 3: We have only enough work for ourselves
#     if length(local_queue) >= 1
#         # @debug "[Thread $(threadid)] Case 3: We have only enough work for ourselves ($(length(local_queue)) items; $(state.needs_work_any))"
#         return false
#     end
#     # Case 4: We have no work but need it
#     if length(local_queue) == 0
#         # @debug "[Thread $(threadid)] Case 4: We have no work but need it"
#         lock(state.lock) do
#             # @debug "[Thread $(threadid)] Updating needs_work"
#             state.needs_work[threadid] = true
#             state.needs_work_any = true
#             state.should_exit = all(state.needs_work) && length(state.common_queue.queue) == 0
#             # @debug "[Thread $(threadid)] Updated needs_work"
#             if state.should_exit
#                 # @debug "[Thread $(threadid)] should_exit=true -> Notifying all threads"
#                 notify(state.condition)
#             end
#         end
#         while !state.should_exit && length(local_queue) == 0
#             # @debug "[Thread $(threadid)] Waiting for work: $(state.needs_work) -> $(state.needs_work_any), $(state.should_exit)"
#             #if length(state.common_queue.queue) > 0
#             lock(state.lock) do
#                 while length(state.common_queue) == 0 && !state.should_exit
#                     wait(state.condition)
#                 end
#                 if length(state.common_queue) > 0
#                     push!(local_queue,pop!(state.common_queue))
#                     state.needs_work[threadid] = false
#                     state.needs_work_any = any(state.needs_work)
#                 end
#             end
#         end
#         return state.should_exit
#     end
#     # @debug "[Thread $(threadid)] Case 5: Uncovered case?"
# end