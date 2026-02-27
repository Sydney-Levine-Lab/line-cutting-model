"""
Extract structural features from PDDL map files in src/maps/.

Uses PDDL.jl (with Arrays extension) to parse map files at the domain level,
avoiding manual regex parsing of the bit-matrix wall representation.

Usage:
  julia --project=src analysis/extract_map_features.jl

Output: data/scenarios/map-features.csv
"""

using PDDL
using CSV
using Printf
using Graphs
using GraphsFlows
using SparseArrays

PDDL.Arrays.@register()

const GRID_SIZE   = 14
const INF_CAP     = 10^9
const NEIGHBORS   = ((0, 1), (0, -1), (1, 0), (-1, 0))

const DOMAIN_FILE = joinpath(@__DIR__, "..", "src", "domain.pddl")
const MAPS_DIR    = joinpath(@__DIR__, "..", "src", "maps")
const OUTPUT_FILE = joinpath(@__DIR__, "..", "data", "scenarios", "map-features.csv")


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

"""Return the set of traversable (x, y) tiles: not a wall, well, or tank."""
function build_traversable(walls::AbstractMatrix, wells_set, tanks_set)
    trav = Set{Tuple{Int,Int}}()
    for x in 1:GRID_SIZE, y in 1:GRID_SIZE
        walls[y, x] && continue        # walls[y, x] = true → wall at column x, row y
        (x, y) in wells_set && continue
        (x, y) in tanks_set && continue
        push!(trav, (x, y))
    end
    return trav
end


# ---------------------------------------------------------------------------
# Min vertex cut via max-flow on a node-split graph
# ---------------------------------------------------------------------------

"""
Compute the min vertex cut between agent starting positions and
well-adjacent traversable tiles.

Node-split construction:
  - Each traversable tile t  →  t_in (node 2i-1) and t_out (node 2i), 1-indexed.
  - t_in → t_out: capacity INF for source tiles (agents cannot be blocked), 1 otherwise.
  - u_out → v_in: capacity INF for each 4-adjacent traversable pair.
  - super_src → s_in: capacity INF for each source (agent) tile s.
  - t_out → super_snk: capacity INF for each sink tile t (well-adjacent traversable tile).

Sinks have capacity 1 so they can be cut: they represent blockable approach tiles,
unlike agent starting positions which cannot be removed.

Max flow is computed via GraphsFlows.maximum_flow (Edmonds-Karp).
"""
function min_vertex_cut(traversable, agent_positions, wells_set)
    sinks = Set{Tuple{Int,Int}}(
        t for t in traversable
        if any(((t[1]+dx, t[2]+dy) in wells_set) for (dx, dy) in NEIGHBORS)
    )
    sources = Set{Tuple{Int,Int}}(p for p in agent_positions if p in traversable)

    (isempty(sinks) || isempty(sources)) && return 0

    tile_list = sort(collect(traversable))
    idx       = Dict(t => i for (i, t) in enumerate(tile_list))
    n         = length(tile_list)
    N         = 2n + 2
    super_src = 2n + 1
    super_snk = 2n + 2

    g        = SimpleDiGraph(N)
    capacity = spzeros(Int, N, N)

    function add_cap!(u, v, c)
        Graphs.add_edge!(g, u, v)
        capacity[u, v] = c
    end

    # Internal edges: t_in → t_out
    for (tile, i) in idx
        add_cap!(2i - 1, 2i, tile in sources ? INF_CAP : 1)
    end

    # Cross edges: u_out → v_in for each 4-adjacent pair
    for (tile, i) in idx
        x, y = tile
        for (dx, dy) in NEIGHBORS
            nb = (x + dx, y + dy)
            haskey(idx, nb) || continue
            add_cap!(2i, 2idx[nb] - 1, INF_CAP)
        end
    end

    # super_src → s_in for each source
    for tile in sources
        add_cap!(super_src, 2idx[tile] - 1, INF_CAP)
    end

    # t_out → super_snk for each sink
    for tile in sinks
        add_cap!(2idx[tile], super_snk, INF_CAP)
    end

    flow, _ = maximum_flow(g, super_src, super_snk, capacity, EdmondsKarpAlgorithm())
    return Int(flow)
end


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

function extract_features(domain, map_path)
    problem = load_problem(map_path)
    state   = initstate(domain, problem)

    # walls[y, x] == true  →  wall at grid position (x, y)
    walls = state[pddl"(walls)"]

    agents = Dict{String, Tuple{Int,Int}}()
    wells  = Dict{String, Tuple{Int,Int}}()
    tanks  = Dict{String, Tuple{Int,Int}}()

    for (obj, typ) in PDDL.get_objtypes(state)
        name = string(obj)
        x    = state[Compound(:xloc, Term[obj])]
        y    = state[Compound(:yloc, Term[obj])]
        if     typ == :agent; agents[name] = (x, y)
        elseif typ == :well;  wells[name]  = (x, y)
        elseif typ == :tank;  tanks[name]  = (x, y)
        end
    end

    wells_set   = Set(values(wells))
    tanks_set   = Set(values(tanks))
    traversable = build_traversable(walls, wells_set, tanks_set)

    n_obstacles = GRID_SIZE^2 - length(traversable)

    n_water_adj = count(traversable) do (x, y)
        any(((x+dx, y+dy) in wells_set) for (dx, dy) in NEIGHBORS)
    end

    bottleneck = min_vertex_cut(traversable, collect(values(agents)), wells_set)
    return (
        map                         = splitext(basename(map_path))[1],
        n_obstacles,
        n_water_adjacent_traversable = n_water_adj,
        bottleneck_size             = bottleneck,
    )
end


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

function main()
    domain    = load_domain(DOMAIN_FILE)
    map_files = sort(filter(endswith(".pddl"), readdir(MAPS_DIR, join=true)))

    rows = Vector{NamedTuple}()
    for map_path in map_files
        name = splitext(basename(map_path))[1]
        print("Processing $name ... ")
        row = extract_features(domain, map_path)
        push!(rows, row)
        @printf("bottleneck=%d, n_obstacles=%d, n_water_adj=%d\n",
                row.bottleneck_size, row.n_obstacles, row.n_water_adjacent_traversable)
    end

    mkpath(dirname(OUTPUT_FILE))
    CSV.write(OUTPUT_FILE, rows)
    println("\nWrote $(length(rows)) rows to $OUTPUT_FILE")
end

main()
