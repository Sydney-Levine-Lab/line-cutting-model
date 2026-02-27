# Utilities for experiment configuration, logging, and reproducibility.

using InteractiveUtils
using Pkg

# Read integer from ENV
function env_int(key::AbstractString, default::Int)
    try
        parse(Int, strip(get(ENV, key, string(default))))
    catch
        default
    end
end

# Read float from ENV
function env_float(key::AbstractString, default::Float64)
    raw = strip(get(ENV, key, string(default)))
    try
        parse(Float64, raw)
    catch
        default
    end
end

# Read boolean flag from ENV
function env_bool(key::AbstractString, default::Bool=false)
    raw = get(ENV, key, "")
    isempty(strip(raw)) && return default
    v = lowercase(strip(raw))
    if v in ("1","true","t","yes","y","on")
        return true
    elseif v in ("0","false","f","no","n","off")
        return false
    else
        return default
    end
end

# Thread-safe println
const PRINT_LOCK = ReentrantLock()
tprintln(msg) = lock(PRINT_LOCK) do
    println(msg)
end

# Write a reproducibility snapshot (git, Julia, dependencies) to output directory
function write_snapshot!(outdir::AbstractString, src_dir::AbstractString)
    mkpath(outdir)

    # git commit
    try
        git_head = readchomp(`git -C $(SRC_DIR) rev-parse HEAD`)
        open(joinpath(outdir, "git_commit.txt"), "w") do io
            println(io, git_head)
        end
    catch
        open(joinpath(outdir, "git_commit.txt"), "w") do io
            println(io, "UNKNOWN (not a git repo or git not available)")
        end
    end

    # Julia version info
    open(joinpath(outdir, "julia_version.txt"), "w") do io
        println(io, "Julia ", VERSION)
        println(io)
        InteractiveUtils.versioninfo(io)
    end

    # dependency status (Project + Manifest)
    open(joinpath(outdir, "pkg_status.txt"), "w") do io
        println(io, "Project dependencies:")
        Pkg.status(; io=io)
        println(io)
        println(io, "Full manifest:")
        Pkg.status(; io=io, mode=Pkg.PKGMODE_MANIFEST)
    end
    
    # copy Project/Manifest used for this run (if present)
    for f in ("Project.toml", "Manifest.toml")
        src = joinpath(SRC_DIR, f)
        if isfile(src)
            cp(src, joinpath(outdir, f); force=true)
        end
    end

    # config receipt
    open(joinpath(outdir, "run_config.txt"), "w") do io
        println(io, "RUN_ID = ", RUN_ID)
        println(io, "RUN_LABEL = ", RUN_LABEL)
        println(io, "N_AGENTS = ", N_AGENTS)
        println(io, "TIME_MAX = ", TIME_MAX)
        println(io, "TEMPERATURE = ", TEMPERATURE)
        println(io, "RUNS = ", RUNS)
        println(io, "SAVE_TRAJECTORIES = ", SAVE_TRAJECTORIES)
        println(io, "VERBOSE = ", VERBOSE)
        println(io, "BASE_SEED = ", BASE_SEED)
        println(io, "MAP_SEED_OFFSET = ", MAP_SEED_OFFSET)
        println(io, "THREADS = ", Threads.nthreads())
        println(io, "MAP_FILES = ", join(MAP_FILES, ", "))
        println(io, "ORDER = ", ORDER)
        println(io, "INFO = ", INFO)
        println(io, "REASONING = ", REASONING)
        println(io, "PLANNING = ", PLANNING)
        println(io, "PLANNING_H_MULT = ", PLANNING_H_MULT)
        println(io, "PLANNING_SEARCH_NOISE = ", PLANNING_SEARCH_NOISE)
    end
end

