using Revise

using POMDPs
# using POMDPModels
# using POMDPModels: LDNormalStateDist, LightDark1DState
using POMDPSimulators
using POMDPModelTools
using ParticleFilters
using POMCPOW
using QMDP
using Statistics
using Random
using SubHunt

# function POMCPOW.estimate_value(o::QMDP.AlphaVectorPolicy{SubHuntPOMDP, Int64},
#                         pomdp::SubHuntPOMDP, s, h::POMCPOW.BeliefNode, steps)
#     b = ParticleCollection{SubState}([s])
#     POMDPs.value(o, b)
# end

# next_action(o, pomdp, b, h)

pomdp = SubHuntPOMDP()

qmdp = QMDPSolver(max_iterations=30,
                    belres=1e-3,
                    verbose=true
                   )
# qmdp_policy = solve(qmdp, pomdp)

ds0 = POMDPs.initialstate_distribution(pomdp)
up = BootstrapFilter(pomdp, 1000)
b0 = POMDPs.initialize_belief(up, ds0)

solver = POMCPOWSolver(tree_queries=100,
                        check_repeat_obs=true,
                        check_repeat_act=true,
                        # estimate_value=POMCPOW.RolloutEstimator(qmdp_policy),
                        # next_action=qmdp_policy,
                        rng=MersenneTwister(1),
                        default_action=1
                        )
planner = POMDPs.solve(solver, pomdp)

Random.seed!(1)
# action_info(planner, b0)
N = 250
rs = RolloutSimulator(max_steps=50)
V = Float64[]
println("Starting simulations")
for i in 1:N
    println("Trial $i")
    v = simulate(rs, pomdp, planner, up, b0)
    push!(V, v)
end
mean_v = mean(V)
se_v = std(V)/sqrt(N)
println("Discounted Return: $mean_v ± $se_v")
