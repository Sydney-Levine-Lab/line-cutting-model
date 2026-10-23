using Test, Random, PDDL, PlanningDomains, SymbolicPlanners

const TARGET_SRC = get(ENV, "LINES_TARGET_SRC", normpath(joinpath(@__DIR__, "..", "src")))
ENV["DEPTH"] = "0"
ENV["LEVEL"] = "0"
ENV["ZAP"] = "false"
ENV["TEMPERATURE"] = "0.0001"
ENV["EVAL_STEPCOST"] = "false"
ENV["TIEBREAK"] = "first"
ENV["WRITE_SNAPSHOT"] = "false"
ENV["TRAJECTORY_LEVEL"] = "none"
ENV["USE_TIMESTAMP"] = "false"
ENV["RUN_LABEL"] = "prediction_timing_test"

module Current
    include(joinpath(Main.TARGET_SRC, "main.jl"))
end
ENV["DEPTH"] = "1"
module ShallowL1
    include(joinpath(Main.TARGET_SRC, "main.jl"))
end
ENV["LEVEL"] = "1"
module ShallowL2
    include(joinpath(Main.TARGET_SRC, "main.jl"))
end
ENV["DEPTH"] = "0"
ENV["LEVEL"] = "0"
module Baseline
    # Optional frozen pre-fix driver; normal tests require no Git access or
    # historical checkout. This extra check was used during patch validation.
    if haskey(ENV, "LINES_PREDICTION_BASELINE")
        source = read(ENV["LINES_PREDICTION_BASELINE"], String)
        Base.include_string(@__MODULE__, source, joinpath(Main.TARGET_SRC, "main.jl"))
    end
end

const DOMAIN = PlanningDomains.load_domain(Current.DOMAIN_FILE)
const MAPS = ["no_line_1.pddl", "no_line_2.pddl", "yes_line_7.pddl",
              "yes_line_8.pddl", "maybe_6.pddl", "yes_line_E.pddl"]

initial_world(map) = initstate(DOMAIN, PlanningDomains.load_problem(joinpath(Current.MAPS_DIR, map)))
function subset_world(map, order)
    Current.remove_agents(initial_world(map), DOMAIN, [k for k in 1:8 if k ∉ order])
end
function queue_world(order=[1, 2, 3])
    state = subset_world("no_line_1.pddl", order)
    for (i, k) in enumerate(order)
        state[Current.XLOC[k]], state[Current.YLOC[k]] = 6 - i, 5
    end
    return state
end

# Test-only one-action policies record exactly which single-agent view was
# offered to the native sampler. They do not replace production policies.
struct ProbePolicy <: PolicySolution
    agent::Int
    seen::Vector{Any}
    rule::Function
end
function SymbolicPlanners.get_action_values(policy::ProbePolicy, state::State)
    k = policy.agent
    push!(policy.seen, (; k, x=state[Current.XLOC[k]], y=state[Current.YLOC[k]],
                       walls=copy(state[pddl"(walls)"]),
                       water=(state[Current.HAS_WATER1[k]], state[Current.HAS_WATER2[k]],
                              state[Current.HAS_WATER3[k]], state[Current.HAS_FILLED[k]])))
    action = policy.rule(state, k)
    @test action in collect(PDDL.available(DOMAIN, state))
    return Dict{Term,Float64}(action => 0.0)
end
function prefer_right(state, k)
    legal = Set(PDDL.available(DOMAIN, state))
    for name in (:right, :up, :down, :left, :wait)
        action = Compound(name, Term[Current.AGENTS[k]])
        action in legal && return action
    end
    return first(legal)
end
function probes(; rule=prefer_right)
    seen = Any[]
    policies = [BoltzmannPolicy(ProbePolicy(k, seen, rule), 0.0001) for k in 1:8]
    return policies, seen
end
act(name, k) = Compound(name, Term[Current.AGENTS[k]])

struct CaptureCost
    views::Vector{Any}
end
function SymbolicPlanners.compute(cost::CaptureCost, domain::Domain, state::State, goal)
    push!(cost.views, copy(state))
    return 17.0
end

# Independent real-round oracle: collect each intention from the untouched
# snapshot, then resolve the collected intentions in priority order.
function real_l0_round(state, policies, order)
    intentions = [get_action(policies[k], Current.project_others_to_walls(state, k, DOMAIN))
                  for k in order]
    execution = state
    for action in intentions
        execution = Current.safe_transition(DOMAIN, execution, action)
    end
    return execution
end

function no_overlap(state, order)
    positions = [(state[Current.XLOC[k]], state[Current.YLOC[k]]) for k in order]
    return length(unique(positions)) == length(order)
end

@testset "prediction information and execution timing" begin
    @testset "L0 intentions use the shared snapshot" begin
        state = queue_world()
        before = copy(state)
        policies, seen = probes()
        actual = Current.simulate_round(state, DOMAIN, policies, [1, 2, 3], 0)
        @test state == before
        @test [(s.k, s.x, s.y) for s in seen] == [(1, 5, 5), (2, 4, 5), (3, 3, 5)]
        @test seen[2].walls[5, 5] # agent 1 still occupies this cell to L0 agent 2
        @test seen[3].walls[5, 4] # agent 2 still occupies this cell to L0 agent 3
        @test [(actual[Current.XLOC[k]], actual[Current.YLOC[k]]) for k in 1:3] ==
              [(6, 5), (4, 4), (3, 4)]
        @test no_overlap(actual, [1, 2, 3])

        policies, seen = probes()
        view = Current.build_planning_state_d1(state, 3, DOMAIN, policies, [1, 2, 3], 3, 0)
        @test view[pddl"(walls)"][5, 6]
        @test view[pddl"(walls)"][4, 4]
        @test !view[pddl"(walls)"][5, 5]
        @test !view[pddl"(walls)"][5, 4]
        @test seen[2].walls[5, 5]
    end

    @testset "priority resolves collisions after intentions are formed" begin
        state = queue_world([1, 2])
        state[Current.XLOC[1]], state[Current.YLOC[1]] = 4, 5
        state[Current.XLOC[2]], state[Current.YLOC[2]] = 6, 5
        collision_rule = (s, k) -> act(k == 1 ? :right : :left, k)
        for order in ([1, 2], [2, 1])
            policies, seen = probes(; rule=collision_rule)
            actual = Current.simulate_round(state, DOMAIN, policies, order, 0)
            winner, loser = order
            @test actual[Current.XLOC[winner]] == 5
            @test actual[Current.XLOC[loser]] == state[Current.XLOC[loser]]
            @test no_overlap(actual, order)
            @test all(!s.walls[5, 5] for s in seen)
        end
    end

    @testset "nested private predictions do not advance prefixes twice" begin
        for order in ([1, 2, 3], [3, 2, 1], [2, 4, 8]), level in 1:3
            state = queue_world(order)
            before = copy(state)
            policies, seen = probes()
            Current.build_planning_state_d1(state, last(order), DOMAIN, policies,
                                            order, length(order), level)
            @test state == before
            # Every modeled actor is still at its own round-start location
            # when it forms an intention, including recursive private models.
            @test all(s.x == state[Current.XLOC[s.k]] &&
                      s.y == state[Current.YLOC[s.k]] for s in seen)

            policies, seen = probes()
            actual = Current.simulate_round(state, DOMAIN, policies, order, level)
            @test state == before
            @test all(s.x == state[Current.XLOC[s.k]] &&
                      s.y == state[Current.YLOC[s.k]] for s in seen)
            @test no_overlap(actual, order)
        end
    end

    @testset "rollout candidates do not become observations of later agents" begin
        state = queue_world()
        for level in (0, 1, 2), idx in 1:3
            k = [1, 2, 3][idx]
            observed = Vector{Any}[]
            for candidate in (act(k == 2 ? :up : :right, k), act(:wait, k))
                policies, seen = probes()
                cost = CaptureCost(Any[])
                before = copy(state)
                @test Current.evaluate_action(state, k, candidate, DOMAIN, policies,
                                               [1, 2, 3], idx, 1, level, cost) == -17.0
                @test state == before
                @test length(cost.views) == 1
                push!(observed, seen)
                if level == 0 && idx == 1
                    @test cost.views[1][pddl"(walls)"][4, 4]
                    @test cost.views[1][pddl"(walls)"][4, 3]
                    @test !cost.views[1][pddl"(walls)"][5, 4]
                elseif level == 0 && idx == 3
                    @test cost.views[1][pddl"(walls)"][5, 6]
                    @test cost.views[1][pddl"(walls)"][4, 4]
                end
            end
            # The forced action changes physical execution, but not the
            # private planning views offered to the other agents in round 1.
            @test observed[1] == observed[2]
        end
    end

    @testset "private prefixes cannot collect multiple water phases in one round" begin
        state = queue_world()
        for (k, x) in ((1, 13), (2, 12), (3, 11))
            state[Current.XLOC[k]] = x
        end
        collect_rule = function (s, k)
            if k == 1
                for action in PDDL.available(DOMAIN, s)
                    action.name in (:collect1, :collect2, :collect3) && return action
                end
            end
            return prefer_right(s, k)
        end
        for level in 1:3
            policies, seen = probes(; rule=collect_rule)
            result = Current.simulate_round(state, DOMAIN, policies, [1, 2, 3], level)
            @test result[Current.HAS_WATER1[1]]
            @test !result[Current.HAS_WATER2[1]]
            @test !result[Current.HAS_WATER3[1]]
            @test all(s.water == (false, false, false, false) for s in seen if s.k == 1)
        end
    end

    @testset "new rounds observe the previous round's resolved world" begin
        state = queue_world()
        policies, seen = probes()
        first_round = Current.simulate_round(state, DOMAIN, policies, [1, 2, 3], 0)
        empty!(seen)
        second_round = Current.simulate_round(first_round, DOMAIN, policies, [1, 2, 3], 0)
        @test [(s.x, s.y) for s in seen] ==
              [(first_round[Current.XLOC[k]], first_round[Current.YLOC[k]]) for k in 1:3]
        @test no_overlap(second_round, [1, 2, 3])

        policies, seen = probes()
        cost = CaptureCost(Any[])
        @test Current.evaluate_action(state, 1, act(:right, 1), DOMAIN, policies,
                                      [1, 2, 3], 1, 2, 0, cost) == -17.0
        # Calls are agents 2,3 for round 1, then agents 1,2,3 for round 2.
        @test [s.k for s in seen] == [2, 3, 1, 2, 3]
        @test [(s.x, s.y) for s in seen[3:5]] == [(6, 5), (4, 4), (3, 4)]
    end

    @testset "native L0 imagined rounds equal the real L0 round on all maps" begin
        for map in Current.CANONICAL_MAP_FILES
            state = initial_world(map)
            before = copy(state)
            policies = Current.build_agent_policies(DOMAIN, Current.build_steps_to_go_estimator())
            for seed in (1, 23, 2026)
                Random.seed!(seed)
                expected = real_l0_round(state, policies, collect(1:8))
                next_random = rand()
                Random.seed!(seed)
                actual = Current.simulate_round(state, DOMAIN, policies, collect(1:8), 0)
                @test actual == expected
                @test rand() == next_random
                @test state == before
                @test no_overlap(actual, collect(1:8))
            end
        end
    end

    @testset "native predictions match actual shallow agents at the modeled level" begin
        for (mod, level, maps) in ((ShallowL1, 1, MAPS), (ShallowL2, 2, MAPS[1:3]))
            for map in maps, order in ([1, 2, 3], [3, 2, 1], [2, 4, 8])
                state = subset_world(map, order)
                policies = mod.build_agent_policies(DOMAIN, mod.build_steps_to_go_estimator())
                for seed in (11, 2026)
                    Random.seed!(seed)
                    expected = mod.simulation_step(state, policies, DOMAIN, order,
                                                   mod.build_steps_to_go_estimator())
                    next_random = rand()
                    Random.seed!(seed)
                    actual = Current.simulate_round(state, DOMAIN, policies, order, level)
                    @test actual == expected
                    @test rand() == next_random
                    @test no_overlap(actual, order)
                end
            end
        end
    end

    if isdefined(Baseline, :simulation_step)
        @testset "frozen driver reproduces the defects caught by these checks" begin
            state = queue_world()
            policies, seen = probes()
            old_round = Baseline.simulate_round(state, DOMAIN, policies, [1, 2, 3], 0)
            @test old_round[Current.XLOC[2]] == 5
            @test old_round[Current.YLOC[2]] == 5
            @test !seen[2].walls[5, 5]

            policies, seen = probes()
            Baseline.build_planning_state_d1(state, 3, DOMAIN, policies, [1, 2, 3], 3, 1)
            @test any(s.x != state[Current.XLOC[s.k]] ||
                      s.y != state[Current.YLOC[s.k]] for s in seen)
        end

        @testset "real L0 behavior and ghosts unchanged from the pre-fix driver" begin
            for map in Current.CANONICAL_MAP_FILES, filled in (false, true)
                original = initial_world(map)
                if filled
                    for k in 1:8
                        original[Current.HAS_FILLED[k]] = true
                        for flag in (Current.HAS_WATER1[k], Current.HAS_WATER2[k], Current.HAS_WATER3[k])
                            original[flag] = false
                        end
                    end
                end
                old_policies = Baseline.build_agent_policies(DOMAIN, Baseline.build_steps_to_go_estimator())
                new_policies = Current.build_agent_policies(DOMAIN, Current.build_steps_to_go_estimator())
                old_state, new_state = original, original
                for step in 1:2
                    Random.seed!(1200 + step)
                    old_state = Baseline.simulation_step(old_state, old_policies, DOMAIN,
                                                         collect(1:8), Baseline.build_steps_to_go_estimator())
                    next_random = rand()
                    Random.seed!(1200 + step)
                    new_state = Current.simulation_step(new_state, new_policies, DOMAIN,
                                                        collect(1:8), Current.build_steps_to_go_estimator())
                    @test old_state == new_state
                    @test rand() == next_random
                end
            end
        end
    end
end
