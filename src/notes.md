# Understanding lvl 1 vs 0: heuristic basically
### Fine, but need to ask Katie?

LEVEL-0 code
- modify_state: is just a helper to treat otehrs as walls. Not the thing that does lvl 0 prediction.
- loop inside code: 
--- select agent
--- modify others to walls
--- act
--- change the state.

LEVEL-1 code
- keep modify_state
- loop inside code:
--- select agent
--- predict! Where are others going to go? Assuming they think that others are walls. 
Hmm.
- act
- etc

-------- CURRENT code
- modify level 1:
for every agent before you,
--- assume that others are walls
--- predict where they will go
--- update the environment

--- returns: state where others are walls,
and have moved according to level 0.
- loop inside code:
--- select agent
--- act based on belief: others will move XXX,
and are now walls. Is this too deterministic? Should I be taking different options into account? (hence the collisions?)

... It's a sort of heuristic for level-1 basically
---- gpt2, stuff to paste into google doc & slack
1. Text to paste into the Google Doc

(New section near the end, after the “New results – Nov 27” part)

Implementation details: level-0 and level-1 models

This section summarizes what the current code is actually doing, in algorithmic terms, for both the level-0 and level-1 variants. It’s meant as a pointer for Katie / Logan / Sydney without requiring them to read the Julia.

Level-0 model: “others as static walls at their current positions”

At level-0, each agent plans as if it were alone in the grid. Other agents are represented only as static obstacles at their current locations. Concretely, for each timestep 
𝑡
t:

We maintain a multi-agent state 
𝑠
𝑡
s
t
	​

 that includes all agents, wells, tanks, walls, etc.

We update agents sequentially in a fixed order 
𝑛
=
1..
𝑁
n=1..N:

For agent 
𝑛
n, we build a single-agent projection of the current multi-agent state by calling modify_state_level_0(s, n). This function:

deletes all other agents from the object list,

records their current positions and turns those positions into walls in the (walls) grid,

keeps all non-agent objects (wells, tanks, finishes, etc.) at their current locations,

preserves agent 
𝑛
n’s internal fluents (has-water1/2/3, has-filled, etc.).
The resulting projected state encodes the level-0 mental model: “I exist, the map exists, and everyone else is just a static wall where they are right now.”

We then query agent 
𝑛
n’s policy on that projected state:

Each agent’s policy is built by composing

WellTankHeuristic
→
A* planner
→
FunctionalVPolicy
→
BoltzmannPolicy
WellTankHeuristic→A* planner→FunctionalVPolicy→BoltzmannPolicy.
Intuitively, the agent uses an optimistic WellTank heuristic to quickly approximate the value of different states, and then selects an action via a softmax (Boltzmann) over those values.

The chosen action is applied to the real multi-agent state 
𝑠
s via the PDDL transition function, updating only that agent’s position and internal fluents. We count collisions when the transition throws an error (e.g., trying to move into a wall).

After looping through all agents, the resulting multi-agent state becomes 
𝑠
𝑡
+
1
s
t+1
	​

.

In short: at level-0, each agent independently plans as if alone in the grid, treating other agents as static obstacles at their current positions; the true environment evolves by sequentially applying these “single-agent” actions.

Level-1 heuristic: one-step anticipation of level-0 agents

The current “level-1” implementation is best described as a one-step level-1 approximation. Conceptually, a level-1 agent should:

“Assume others are level-0 agents, and choose a best response to that belief.”

In our implementation, for each agent 
𝑘
k at a timestep, we do the following:

Predict earlier agents’ next moves under level-0.

We start from the multi-agent state at the beginning of the timestep, 
𝑠
𝑡
s
t
	​

.

For each earlier agent 
𝑛
=
1..
𝑘
−
1
n=1..k−1:

We construct that agent’s level-0 projected state by calling modify_state_level_0(predicted_state, n), where predicted_state is a simulated world we are updating only inside this prediction procedure.

We query agent 
𝑛
n’s Boltzmann policy on this projected state to obtain a predicted action under level-0.

We apply that predicted action to predicted_state using the PDDL transition function.

After looping over 
𝑛
=
1..
𝑘
−
1
n=1..k−1, we end up with a predicted world in which earlier agents have taken one level-0 step each, in order.

Plan for agent 
𝑘
k in the predicted world.

Starting from this predicted_state, we again call modify_state_level_0(predicted_state, k) to build a single-agent projection for agent 
𝑘
k.

In this projected state, other agents appear as static walls at their predicted next positions (rather than at their current positions).

We then query agent 
𝑘
k’s Boltzmann policy on this projected state to choose its action.

Apply agent 
𝑘
k’s action to the real environment.

The chosen action is applied to the actual multi-agent state 
𝑠
s (not predicted_state), updating the true world.

Thus, a level-1 agent effectively says:

“I expect earlier agents to each take one step according to their level-0 policies. I predict where that will put them, and then I plan my move as if they were static walls at those predicted locations.”

This is strictly richer than level-0 (which uses others’ current locations) but still fairly local: only one future step of others’ behavior is modeled, and once those next positions are predicted, others are treated as static obstacles for planning.

Limitations and possible extensions of the level-1 implementation

The current level-1 implementation has several important limitations:

One-step prediction only.
We only simulate a single future move for each earlier agent; we do not model full multi-step trajectories under their level-0 policies. A richer level-1 would simulate others’ behavior over multiple timesteps and treat their predicted trajectories (not just their next positions) as constraints during planning.

Partial integration of observations.
Prediction for agent 
𝑘
k is based on the state at the start of the timestep and a simulated one-step rollout of earlier agents. In the real environment, earlier agents are also moving sequentially. We are not yet fully combining observed moves and predicted future moves into a unified belief state.

No explicit separation of π⁰ vs π¹ policies.
We use the same Boltzmann policy objects both to simulate other agents’ level-0 behavior and to choose the focal agent’s own action. Conceptually, level-1 is implemented as a different state-transformation (different “mental model of the world”), not as a different policy class.

These limitations are not bugs so much as design choices: they make the level-1 model computationally manageable while still adding a genuine anticipatory component. For clarity, throughout this document I refer to this variant as a “one-step level-1 heuristic” to distinguish it from a fully dynamic level-1 that simulates others’ policies over longer horizons.

2. “Since Nov 27” mini-update

(You can drop this near the top or bottom, as a short dated section)

Update since Nov 27

Since the Nov 27 write-up, I have:

Clarified and documented the implementation of the level-1 model.
I now describe the current variant explicitly as a one-step level-1 heuristic and have written up the algorithm in more detail (see “Implementation details” section). This should make it easier to reason about what the model is doing and how to extend it.

(Planned / in progress) Hooking up participant data.
I am wiring the existing analysis pipeline to the participant data that Logan shared, so that we can directly compare per-map behavior of the level-0 and level-1 models to human responses (e.g., universalizability judgments). My aim is to have first-pass plots of “humans vs model (L0 vs L1)” per map before our Dec 17 meeting.

Exploring conceptual next steps for richer level-1.
I have also been thinking through what a more fully dynamic level-1 model would look like (e.g., simulating multi-step trajectories of level-0 agents, integrating observed moves and predicted future moves). I’ve sketched these options so we can decide together how ambitious we want to be for the January deadline.

If you want to be extra precise, you can edit the “(Planned / in progress)” bullet once you actually have the plots.

3. “Questions for Dec 17 meeting” box
Questions for the Dec 17 meeting

Level-1 model scope.
Are we comfortable treating the current one-step level-1 heuristic (agents predict earlier agents’ next level-0 move, then plan around those predicted positions) as the main “level-1” variant for the CogSci / January timeline, or do we want to invest in a richer, multi-step level-1 model?

Model–data comparison priorities.
Once basic “humans vs model” plots are in place, which aspects should we prioritize: overall fit (e.g., correlation across maps), particular “funnel” maps where L0 and L1 differ, or specific patterns in the cross-cultural data?

Time commitment / scope after January.
Assuming things go well, how much of my time between January and April would you like to devote to:
(a) refining the lines model (e.g., richer level-1),
(b) spinning out an additional project (e.g., evolutionary game-theoretic modeling of norms), or
(c) some mix of both?

4. Slack message to send in #igm-lines

Here’s something you can paste almost verbatim (tweak tone if you like). This assumes you’ll add the new section to the doc first or right after:

Hi all! Quick update before our Dec 17 meeting:

– I’ve clarified and documented the implementation of the “level-1” model in our Google doc. The current variant is best described as a one-step level-1 heuristic: for each agent, I simulate earlier agents taking one level-0 step, then plan as if those predicted next positions were static walls. There’s now a short “Implementation details” section that spells out the level-0 and level-1 algorithms.

– I’m in the process of wiring the existing analysis pipeline to the participant data Logan shared, so we can compare per-map behavior of the level-0 vs level-1 models to human responses (e.g., universalizability judgments). My goal is to have first-pass “humans vs model” plots in the doc before the 17th.

I also added a small list of questions for the Dec 17 meeting (e.g., how ambitious we want to be about a richer, multi-step level-1). If there’s anything in particular you’d like me to prioritize between now and then, I’m very happy to adjust.

You can send this once you’ve pasted the new section into the doc (even if the participant-data part is “in progress” — just be honest about that).

If you want, next we can sketch the simplest possible humans-vs-model plot (what metric, what axes) so that step feels very concrete when you sit down to code it.


---- gpt
Agent’s level-0 belief model:
modified_state = modify_state(current_multi_agent_state, k)
→ others don’t move, they are just walls where they stand.

Agent’s level-0 planning on that belief model:
act = get_action(boltzmann_policy_k, modified_state)
→ plan a path to get water, tank, finish as if in a single-agent grid with those obstacles.

Environment update:
interim_state = transition(domain, interim_state, act)
→ apply that chosen action to the true multi-agent world; collisions and interference happen here, not in the planner.

So yes: modify_state is a helper, but it is the helper that implements the level-0 mental model “others = static obstacles at current locations”.

3. Subtle detail: using interim_state

One small but important nuance in your code:

interim_state = state
for n in 1:N
    modified_state = modify_state(interim_state, n)
    act = SymbolicPlanners.get_action(boltzmann_policies[n], modified_state)
    interim_state = transition(domain, interim_state, act)
end
state = interim_state


At timestep t:

Agent 1 sees interim_state = state_t (everyone still in their positions at start of timestep), projects others to walls, plans, moves.

Agent 2 now sees updated interim_state where agent 1 has already moved this timestep.
It projects that to walls and plans.

And so on…

So within a single timestep, agents move sequentially and each later agent is (level-0) responding to earlier agents’ actual new positions, but still treating them as if they will stay there forever.

That’s a nice hybrid:

No prediction of future movement (“they’ll step aside later”)

But yes to noticing where earlier agents currently ended up on this timestep.

Still purely level-0: nobody simulates another agent’s future plans or beliefs; they just treat the latest snapshot of everyone as static obstacles.