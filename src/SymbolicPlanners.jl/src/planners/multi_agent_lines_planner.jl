#setup
export MultiAgentRealTimeDynamicPlanner, MultiAgentRTDP

struct MultiAgentDomain{D <: Domain} <: Domain
    domain::D
end

n_agents = 8
max_timesteps = 500

"Multi-agent planner that uses Real Time Dynamic Programming (RTDP)."
@kwdef mutable struct MultiAgentRealTimeDynamicPlanner <: Planner
    #heuristic::Heuristic = GoalCountHeuristic()
    heuristic::Heuristic = WellTankHeuristic()
    n_rollouts::Int = 50
    max_depth::Int = 50
    rollout_noise::Float64 = 0.0
    action_noise::Float64 = 0.0
end

const MultiAgentRTDP = MultiAgentRealTimeDynamicPlanner


#transition domain to roll out action and check for collisions
function PDDL.transition(domain::MultiAgentDomain, state::State, actions)
    agent_locs = []
    # TODO: order the actions first? pddl"(move agent1 left)"

    #basically, right now we iterate through the list of agents in the same order each time setup
    for (i, act) in enumerate(actions)
	    agent = Const(Symbol("agent$i"))
        next_state = transition(domain.domain, state, act)
        next_agent_loc = (next_state[Compound(:xloc, Term[agent])], next_state[Compound(:yloc, Term[agent])])

        #if an agent tries to move into a square that a preceding agent has moved into, then this agent stays still
	    if next_agent_loc in agent_locs
	        agent_loc = (state[Compound(:xloc, Term[agent])], state[Compound(:yloc, Term[agent])])
            push!(agent_locs, agent_loc)
        #no collision, so move the agent into the new location
        else
            state = next_state
            push!(agent_locs, next_agent_loc)
        end
    end
    return state
end

# #load specific domain and problem
# domain_path = "/Users/joekwon/Desktop/line_modeling"
# multi_agent_lines = load_domain(joinpath(domain_path, "domain.pddl"))
# problem_path = "/Users/joekwon/Desktop/line_modeling/2_agent_maps"
# mal_problem = load_problem(joinpath(problem_path, "no_line_3.pddl"))

# PDDL.Arrays.@register()


# state = initstate(multi_agent_lines, mal_problem)
# multi_domain = MultiAgentDomain(domain)

# # Simulate all k=0 agents
# agent_policies = []

# for t in 1:max_timesteps
#     actions = Term[]
#     for agent_id in 1:n_agents
#         policy = agent_policies[agent_id]
#         act = get_action(policy, state)
#         push!(actions, act)
#     end
#     state = PDDL.transition(multi_domain, state, actions)
# end

function solve(planner::MultiAgentRTDP,
               domain::Domain, state::State, spec::Specification)
    # Initialize then refine solution
    default = FunctionalVPolicy(planner.heuristic, domain, spec)
    sol = TabularPolicy(default=default)
    sol.V[hash(state)] = -planner.heuristic(domain, state, spec)
    sol = solve!(planner, sol, domain, state, spec)
    # Wrap in Boltzmann policy if needed
    return planner.action_noise == 0 ?
        sol : BoltzmannPolicy(sol, planner.action_noise)
end

#you can plan over the single agent domain with A* or BoltzmannPolicy as long as agents are fixed
#or you can plan over multi agent where people are moving. 

function solve!(planner::MultiAgentRTDP, sol::TabularPolicy,
                domain::Domain, state::State, spec::Specification)
    @unpack heuristic, action_noise = planner
    @unpack n_rollouts, max_depth, rollout_noise = planner
    # Precompute heuristic information
    precompute!(heuristic, domain, state, spec)
    # Perform rollouts from initial state
    initial_state = state
    ro_policy = rollout_noise == 0 ? sol : BoltzmannPolicy(sol, rollout_noise)
    visited = Vector{typeof(state)}()
    for n in 1:n_rollouts
        state = initial_state
        # Rollout until maximum depth
        for t in 1:max_depth
            push!(visited, state)
            if is_goal(spec, domain, state) break end
            update_values!(planner, sol, domain, state, spec)
            # Get actions according to policy for each agent
            actions = Term[]
            for agent_id in 1:n_agents
                policy = agent_policies[agent_id]
                act = get_action(policy, state)
                push!(actions, act)
            end
            # Apply transition function for each action
            for act in actions
                state = transition(domain, state, act)
            end
        end
        # Post-rollout update
        while length(visited) > 0
            state = pop!(visited)
            update_values!(planner, sol, domain, state, spec)
        end
    end
    return sol
end


function update_values!(planner::MultiAgentRTDP, sol::TabularPolicy,
                        domain::Domain, state::State, spec::Specification)
    # TODO: Iterate over a reasonable subset of all possible actions instead

    # look at and modify available function. need to do available over all agents separately
    # one idea is random sample of available actions
    actions = collect(available(domain, state))
    state_id = hash(state)
    if is_goal(spec, domain, state) #need to modify is_goal
        qs = zeros(length(actions))
        sol.Q[state_id] = Dict{Term,Float64}(a => 0 for a in actions)
        sol.V[state_id] = 0.0
        return
    end
    qs = map(actions) do act #don't forget that right now this is operating on all 8 agents' actions, but need to look at sets of actions (for the 8 agents)
        next_state = transition(domain, state, act)
        next_v = get!(sol.V, hash(next_state)) do
            -planner.heuristic(domain, next_state, spec)
        end
        r = get_reward(spec, domain, state, act, next_state)
        return get_discount(spec) * next_v + r
    end
    sol.Q[state_id] = Dict{Term,Float64}(zip(actions, qs)) #need to initialize Q value dictionary as some default dict. need to install datastruct package for DefaultDict?
    sol.V[state_id] = planner.action_noise == 0 ? 
        maximum(qs) : sum(softmax(qs ./ planner.action_noise) .* qs)
end


#k=1
struct Level1Domain <: Domain
    multi_agent_domain::MultiAgentDomain
    other_policies::Vector
end

function PDDL.transition(domain::Level1Domain, state::State, action)
    other_actions = [get_action(p, state) for p in domain.other_policies]
    #TODO: splice my action into other_actions at the right index
    #actions = 
    state = PDDL.transition(domain.multi_agent_domain, state, actions)
end