using Revise

using POMDPs
# using POMDPModels
# using POMDPModels: LDNormalStateDist, LightDark1DState
using POMDPSimulators
# using POMDPModelTools
using ParticleFilters
using POMCPOW
using QMDP
using Statistics
using Random
using SubHunt

function POMCPOW.estimate_value(o::QMDP.AlphaVectorPolicy{SubHuntPOMDP, Int64},
                        pomdp::SubHuntPOMDP, s, h::POMCPOW.BeliefNode, steps)
    b = ParticleCollection{SubState}([s])
    POMDPs.value(o, b)*0.5
end

# next_action(o, pomdp, b, h)

# function estimate_value(p::SubHuntPOMDP, s::SubState, h, steps)
#     d = sum(abs.(s0.own - s0.target))
#     steps = d/3
#     hit_rate = s.aware ? 0.6 : 1.0
#     v = POMDPs.discount(p)^steps*100.0*hit_rate*0.1
#     return v
# end

pomdp = SubHuntPOMDP()

qmdp = QMDPSolver(max_iterations=30,
                    belres=1e-3,
                    verbose=true
                   )
qmdp_policy = solve(qmdp, pomdp)

ds0 = POMDPs.initialstate_distribution(pomdp)
up = BootstrapFilter(pomdp, 10000)


solver = POMCPOWSolver(tree_queries=100,
                        # check_repeat_obs=true,
                        check_repeat_act=true,
                        # estimate_value=estimate_value,
                        estimate_value=qmdp_policy,
                        # next_action=qmdp_policy,
                        k_action=10,
                        alpha_action=0.5,
                        k_observation=3,
                        alpha_observation=0.2,
                        rng=MersenneTwister(1),
                        default_action=1
                        )
planner = POMDPs.solve(solver, pomdp)

Random.seed!(1)
# action_info(planner, b0)
N = 1000
rs = RolloutSimulator(max_steps=50)
V = Float64[]
println("Starting simulations")
for i in 1:N
    if (i%25) == 0
        println("Trial $i")
    end
    s0 = rand(ds0)
    b0 = POMDPs.initialize_belief(up, ds0)
    v = simulate(rs, pomdp, planner, up, b0, s0)
    push!(V, v)
end
mean_v = mean(V)
se_v = std(V)/sqrt(N)
println("Discounted Return: $mean_v ± $se_v")
