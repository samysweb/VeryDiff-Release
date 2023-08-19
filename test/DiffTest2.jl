using TimerOutputs
using LinearAlgebra

using VeryDiff
using VNNLib

# This is still missing the optimizations for instable + linear neurons
function run()
    reset_timer!(VeryDiff.to)
    N_original = parse_network(VNNLib.load_network("./test/examples/networks/acas-2-7.onnx"))
    #N_original = parse_network(VNNLib.load_network("./test/examples/networks/acas-1-3.onnx"))

    N32 = VeryDiff.truncate_network(Float32,N_original)
    N16 = VeryDiff.truncate_network(Float16,N_original)

    low =  Float64[0.6, -0.5, -0.5, 0.45, -0.5]
    high = Float64[0.679857769, 0.5, 0.5, 0.5, -0.45]

    verify_network(N32,N16,[low high], 0.01) #0.0001) #0.05)
    #               1 Thread    12 Threads
    # 2-7 0.01:     721s        333s
end

#run()