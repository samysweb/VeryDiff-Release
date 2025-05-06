@info "Building Sysimage..."
using PackageCompiler
deps_dir = @__DIR__
create_app("$deps_dir/../", "VeryDiff", precompile_execution_file="$deps_dir/sysimage/trace_run.jl",executables= ["VeryDiff" => "main_VeryDiff"],incremental=true,force=true)