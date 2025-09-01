# Environment setup
- Do you know what Julia version and local/global package versions you used?
- I got things working by getting a recent version of Julia (1.9.4), and copying the WallTankHeuristic directly in the main_modeling_line to avoid issues (it kept using the global version of SymbolicPlanners. Perhaps unrelated, but I think the local version might be missing some files---e.g., boltzmann_policy.jl?) It seems to be working fine as is though, so perhaps this is ok.
# Maps
- Do you have a way to identify which map names correspond to the map numbers used in your paper? And to tell when a map is duplicated? (For package compatibility issues I temporarily shut off PDDLViz, dunno if that's related)
- If I'm not mistaken, in all_data/paper_data, you have two different runs --- one giving a csv per map, and one where maps are collated in order of "all_problems" (data.csv).
- I didn't see such a simple correspondance when comparing with the paper --- like I think no_line_test.pddl is map 26, and yes_line_10_test.pddl is map 1 and map 3.
# Data analysis
- Do you have a script you can share? Using the csv files to make your statistical analyses. Anything that can help me get started, even though this part won't be the hardest :).
# Level 1 reasoning
I'm planning to implement Level 1 reasoning, where, e.g., agents predict others' next move instead of treating them as walls before moving simultaneously. Just curious if you had any thoughts about that / if you had already thought about it (I see you have a commented script to handle collisions, which might become an issue then).