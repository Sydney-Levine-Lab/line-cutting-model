using PDDL, PlanningDomains, SymbolicPlanners, CSV, Random, Statistics, Printf, SHA

# Load definitions, never run a simulated population or write production outputs.
const BASELINE_SRC = get(ENV, "LINES_BASELINE_SRC",
    normpath(joinpath(@__DIR__, "..", "..", "cogsci-2026", "src")))
const SIMULATION_SRC = get(ENV, "LINES_SIMULATION_SRC",
    normpath(joinpath(@__DIR__, "..", "src")))
const LEGACY_HEURISTIC = get(ENV, "LINES_LEGACY_HEURISTIC",
    joinpath(BASELINE_SRC, "water_collection_heuristic.jl"))
const FIXED_HEURISTIC = get(ENV, "LINES_FIXED_HEURISTIC",
    joinpath(@__DIR__, "..", "src", "water_collection_heuristic.jl"))
const BENCH_OUTPUT = get(ENV, "LINES_BENCH_OUTPUT", joinpath(@__DIR__, "..", "outputs", "heuristic_benchmark"))
const REPEATS = parse(Int, get(ENV, "LINES_BENCH_REPEATS", "3"))
const REUSE_ROUTES = lowercase(get(ENV, "LINES_BENCH_REUSE_ROUTES", "false")) == "true"

ENV["ZAP"] = "false"
ENV["TEMPERATURE"] = "0.0001"
ENV["EVAL_STEPCOST"] = "true"
ENV["TIEBREAK"] = "random"
ENV["WRITE_SNAPSHOT"] = "false"
ENV["TRAJECTORY_LEVEL"] = "none"

module Baseline
# Only substitute the legacy helper; all planner and projection definitions
# remain the simulation source verbatim. This also supports a captured baseline
# helper when the archived worktree is unavailable to the current process.
include(Main.LEGACY_HEURISTIC)
main_path = joinpath(Main.SIMULATION_SRC, "main.jl")
main_source = read(main_path, String)
helper_include = "include(joinpath(@__DIR__, \"water_collection_heuristic.jl\"))"
occursin(helper_include, main_source) || error("Cannot locate the heuristic include")
main_source = replace(main_source, helper_include => "")
Base.include_string(@__MODULE__, main_source, main_path)
end

module Corrected
include(Main.FIXED_HEURISTIC)
end

function make_planner(method)
    if method == "legacy"
        return AStarPlanner(Baseline.WaterCollectionHeuristic(); save_search=true)
    elseif method == "corrected"
        return AStarPlanner(Corrected.WaterCollectionHeuristic(); save_search=true)
    elseif method == "uniform"
        return UniformCostPlanner(; save_search=true)
    end
    error("Unknown method $method")
end

function route_cost(sol)
    return sol.status == :success ? Float64(length(sol.plan)) : Inf
end

function make_queries(domain)
    queries = NamedTuple[]
    for map_file in Baseline.CANONICAL_MAP_FILES
        world = initstate(domain, PlanningDomains.load_problem(joinpath(SIMULATION_SRC, "maps", map_file)))
        for k in (1, 8), phase in ("empty", "loaded")
            view = Baseline.project_others_to_walls(world, k, domain)
            view[Baseline.HAS_WATER1[k]] = false
            view[Baseline.HAS_WATER2[k]] = false
            view[Baseline.HAS_WATER3[k]] = phase == "loaded"
            view[Baseline.HAS_FILLED[k]] = false
            goal = MinStepsGoal(Term[Baseline.HAS_FILLED[k]])
            push!(queries, (; id=length(queries)+1, map=replace(map_file, ".pddl"=>""),
                            agent=k, phase, state=view, goal,
                            named_finish=!isempty(PDDL.get_objects(view, :finish))))
        end
    end
    return queries
end

function timed_solve(planner, domain, query)
    measurement = @timed planner(domain, query.state, query.goal)
    sol = measurement.value
    return (; elapsed_seconds=measurement.time, allocated_bytes=measurement.bytes,
            cost=route_cost(sol), expanded=sol.expanded, status=string(sol.status))
end

function benchmark_routes(domain, queries)
    # Warm all method-specific Julia paths before timing any query. The warm-up
    # does not populate any measured planner's cache or precomputed distances.
    for method in ("uniform", "legacy", "corrected")
        for query in queries[1:4]
            make_planner(method)(domain, query.state, query.goal)
        end
    end

    rows = NamedTuple[]
    for (idx, query) in enumerate(queries)
        methods = circshift(["legacy", "corrected", "uniform"], mod(idx, 3))
        for method in methods
            # Cold means a fresh planner/heuristic for each solve. Include its
            # construction and precomputation, but not package/JIT startup.
            for repeat in 1:REPEATS
                measurement = @timed begin
                    planner = make_planner(method)
                    planner(domain, query.state, query.goal)
                end
                sol = measurement.value
                push!(rows, (; id=query.id, map=query.map, agent=query.agent,
                    phase=query.phase, named_finish=query.named_finish,
                    mode="cold", method, repeat,
                    elapsed_seconds=measurement.time, allocated_bytes=measurement.bytes,
                    cost=route_cost(sol), expanded=sol.expanded, status=string(sol.status)))
            end
            # Warm means reusing a heuristic for the same geometry, while still
            # running actual A* every time. No memoized plan answers are used.
            planner = make_planner(method)
            planner(domain, query.state, query.goal)
            for repeat in 1:REPEATS
                result = timed_solve(planner, domain, query)
                push!(rows, (; id=query.id, map=query.map, agent=query.agent,
                    phase=query.phase, named_finish=query.named_finish,
                    mode="warm", method, repeat, result...))
            end
        end
        idx % 14 == 0 && println("Route benchmark: $idx / $(length(queries)) queries")
        flush(stdout)
    end
    return rows
end

function summarize_routes(rows, queries)
    summaries = NamedTuple[]
    for query in queries, mode in ("cold", "warm"), method in ("legacy", "corrected", "uniform")
        group = [r for r in rows if r.id == query.id && r.mode == mode && r.method == method]
        oracle = first(r.cost for r in rows if r.id == query.id && r.method == "uniform")
        costs = unique(r.cost for r in group)
        length(costs) == 1 || error("Nondeterministic route costs for query $(query.id)")
        cost = only(costs)
        if method == "corrected" && cost != oracle
            error("Corrected route mismatch: $(query.map) agent $(query.agent) $(query.phase), $cost versus $oracle")
        end
        push!(summaries, (; id=query.id, map=query.map, agent=query.agent, phase=query.phase,
            named_finish=query.named_finish, mode, method,
            median_seconds=median(r.elapsed_seconds for r in group),
            median_allocated_bytes=median(r.allocated_bytes for r in group),
            expanded=first(group).expanded, cost, oracle_cost=oracle, exact=cost == oracle))
    end
    return summaries
end

function decision_benchmark(domain, queries)
    # Cold-cache cost of computing all immediate-action values, as the actual
    # Boltzmann policy does, on a compact panel spanning finish and congestion.
    panel = Set(["no_line_1", "yes_line_7", "yes_line_8", "maybe_6", "new_maybe_3", "yes_line_E"])
    chosen = [q for q in queries if q.map in panel]
    rows = NamedTuple[]
    for method in ("legacy", "corrected")
        planner = make_planner(method)
        estimator = SymbolicPlanners.memoized(PlannerHeuristic(planner))
        value_policy = FunctionalVPolicy(estimator, domain, first(chosen).goal)
        collect(SymbolicPlanners.get_action_values(value_policy, first(chosen).state))
    end
    for (idx, query) in enumerate(chosen)
        for method in circshift(["legacy", "corrected"], mod(idx, 2)), repeat in 1:REPEATS
            measurement = @timed begin
                planner = make_planner(method)
                estimator = SymbolicPlanners.memoized(PlannerHeuristic(planner))
                value_policy = FunctionalVPolicy(estimator, domain, query.goal)
                collect(SymbolicPlanners.get_action_values(value_policy, query.state))
            end
            push!(rows, (; id=query.id, map=query.map, agent=query.agent, phase=query.phase,
                method, repeat, elapsed_seconds=measurement.time,
                allocated_bytes=measurement.bytes, candidate_count=length(measurement.value)))
        end
    end
    return rows
end

function write_receipt(summaries, decisions, queries)
    open(joinpath(BENCH_OUTPUT, "benchmark_receipt.md"), "w") do io
        println(io, "# Route-planning benchmark\n")
        println(io, "Julia ", VERSION, "; pinned source environment; ", Threads.nthreads(), " thread(s).")
        println(io, "\n", length(queries), " fixed queries: all 28 maps, agents 1 and 8, empty and loaded phases at initial positions. Other agents retain snapshot occupancy as projected walls. No population trajectory or ghost behavior was changed.")
        println(io, "\nEach result is a median of ", REPEATS, " repetitions. JIT warm-up is excluded. Cold trials include fresh planner/heuristic construction and precomputation. Warm trials reuse the same geometry's guide but execute search again. No cached completed-route answers are reused. Method order rotates across queries. Search records are enabled equally for all methods.")
        if REUSE_ROUTES
            println(io, "\nExecution note: completed route measurements were reused after fixing a missing namespace qualification in the benchmark harness's action-value panel. That remaining panel completed in a separate process, also after JIT warm-up. The heuristic source was unchanged. Reuse mode is only valid with the same source, queries, and repetition settings.")
        end
        println(io, "\n| Query group | Mode | Legacy seconds | Corrected seconds | Legacy / corrected | Legacy expansions | Corrected expansions |")
        println(io, "|---|---|---:|---:|---:|---:|---:|")
        for scope in ("all", "no named finish", "named finish", "reachable", "unreachable"), mode in ("cold", "warm")
            group = [r for r in summaries if r.mode == mode &&
                (scope == "all" ||
                 scope == "named finish" && r.named_finish ||
                 scope == "no named finish" && !r.named_finish ||
                 scope == "reachable" && isfinite(r.oracle_cost) ||
                 scope == "unreachable" && !isfinite(r.oracle_cost))]
            isempty(group) && continue
            old = [r for r in group if r.method == "legacy"]
            new = [r for r in group if r.method == "corrected"]
            ot = sum(r.median_seconds for r in old)
            nt = sum(r.median_seconds for r in new)
            @printf(io, "| %s | %s | %.6f | %.6f | %.3f | %d | %d |\n", scope, mode,
                ot, nt, ot/nt, sum(r.expanded for r in old), sum(r.expanded for r in new))
        end
        new = [r for r in summaries if r.method == "corrected" && r.mode == "cold"]
        old = [r for r in summaries if r.method == "legacy" && r.mode == "cold"]
        println(io, "\nCorrected cost agrees with uniform-cost search on ", count(r.exact for r in new), "/", length(new), " queries. Legacy differs on ", count(!r.exact for r in old), " queries. Uniform-cost search found ", count(!isfinite(r.oracle_cost) for r in new), " unreachable queries in this panel; any such query is retained as an Inf cost. Unreachable-state behavior is also covered by the separate route-validation suite.")
        ratios = Float64[]
        for query in queries
            o = only(r for r in old if r.id == query.id)
            n = only(r for r in new if r.id == query.id)
            push!(ratios, o.median_seconds/n.median_seconds)
        end
        println(io, "\nCold route-query speed ratios (legacy/corrected): median ", median(ratios), ", min ", minimum(ratios), ", max ", maximum(ratios), ". Corrected faster on ", count(>(1), ratios), "/", length(ratios), " queries.")
        times = Dict(method => sum(median(r.elapsed_seconds for r in decisions if r.id == query.id && r.method == method)
            for query in queries if any(r.id == query.id for r in decisions)) for method in ("legacy", "corrected"))
        println(io, "\nCold full-action-value panel: ", length(unique(r.id for r in decisions)), " decisions across 6 maps. Sum of decision medians: legacy ", times["legacy"], "s, corrected ", times["corrected"], "s; ratio ", times["legacy"]/times["corrected"], ".")
        println(io, "\nLimits: a fixed route-query panel, not end-to-end model runtime or corrected model fits. Population trajectories can change after route correction. Named-finish legacy routes can be inaccurate, so speed there compares actual implementations rather than equal-quality algorithms. Wall times are machine-specific; expanded counts and raw results are also saved. The API's expanded counter includes its initial count, equally for all methods.")
        println(io, "\nSimulation definitions/maps: `", SIMULATION_SRC, "`. Legacy helper: `", LEGACY_HEURISTIC, "` (SHA-256 `", bytes2hex(sha256(read(LEGACY_HEURISTIC))), "`). Corrected heuristic SHA-256: `", bytes2hex(sha256(read(FIXED_HEURISTIC))), "`.")
    end
end

mkpath(BENCH_OUTPUT)
domain = PlanningDomains.load_domain(joinpath(SIMULATION_SRC, "domain.pddl"))
queries = make_queries(domain)
println("Benchmarking $(length(queries)) matched route queries, $REPEATS repetitions, ghost snapshot occupancy retained.")
flush(stdout)
summaries = if REUSE_ROUTES
    summary_path = joinpath(BENCH_OUTPUT, "benchmark_routes_summary.csv")
    saved = collect(CSV.File(summary_path))
    length(unique(r.id for r in saved)) == length(queries) ||
        error("Saved route results do not match the current query count")
    all(r.exact for r in saved if r.method == "corrected") ||
        error("Saved corrected route results contain a mismatch")
    println("Reusing completed route measurements from $summary_path")
    saved
else
    rows = benchmark_routes(domain, queries)
    summary = summarize_routes(rows, queries)
    CSV.write(joinpath(BENCH_OUTPUT, "benchmark_routes_raw.csv"), rows)
    CSV.write(joinpath(BENCH_OUTPUT, "benchmark_routes_summary.csv"), summary)
    summary
end
decisions = decision_benchmark(domain, queries)
CSV.write(joinpath(BENCH_OUTPUT, "benchmark_decisions_raw.csv"), decisions)
write_receipt(summaries, decisions, queries)
println(read(joinpath(BENCH_OUTPUT, "benchmark_receipt.md"), String))
