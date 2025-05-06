# VeryDiff

VeryDiff is a tool for the equivalence verification of neural networks (NNs).
Given two NNs and a specification of an input region, it can answer the following question:

**Do the two NNs behave *equivalently* on the given input space?**

So far VeryDiff supports three different kinds of equivalence that can be checked for two NNs:
- $\varepsilon$-equivalence: The numerical outputs of the two NNs vary at most by $\varepsilon$ w.r.t. the $L_\infty$-norm
- Top-1 equivalence: The two NNs provide the same classification outputs
- $\delta$-Top-1 equivalence: If for some input the first NN provides classification $c$ with confidence larger $\delta > 0.5$, then the second NN also yields classification $c$.


The guarantees provided by VeryDiff are **sound**, i.e. if VeryDiff says two NNs are equivalent, they are provably so.
On the other hand, in some cases VeryDiff cannot be complete.
However, VeryDiff always tries to find counterexamples and outputs them if found.

## Installation
This software requires Julia 1.10.

Subsequently the software can be installed as follows:

```
git clone https://github.com/samysweb/VeryDiff-Release
cd VeryDiff-Release
./build.sh <path to julia binary>
```

On Linux `<path to julia binary>` can be found via `$(which julia)`.

## Running the tool

The binary of the tool can then be found in `./VeryDiff-Release/deps/VeryDiff/bin/VeryDiff`

Manual:
```
usage: <PROGRAM> [--epsilon EPSILON] [--top-1]
                 [--top-1-delta TOP-1-DELTA] [--timeout TIMEOUT]
                 [--naive] [-h] net1 net2 spec

positional arguments:
  net1                  First NN file (ONNX)
  net2                  Second NN file (ONNX)
  spec                  Input specification file (VNNLIB)

optional arguments:
  --epsilon EPSILON     Verify Epsilon Equivalence; provides the
                        epsilon value (type: Float64, default: -Inf)
  --top-1               Verify Top-1 Equivalence
  --top-1-delta TOP-1-DELTA
                        Verify δ-Top-1 Equivalence; provides the delta
                        value (type: Float64, default: -Inf)
  --timeout TIMEOUT     Timeout for verification (type: Int64,
                        default: 0)
  --naive               Use naive verification (without differential
                        verification)
  -h, --help            show this help message and exit
```