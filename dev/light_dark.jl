using Revise

using POMDPs
using POMDPModels
using POMDPModels: LDNormalStateDist, LightDark1DState
using POMDPSimulators
using ParticleFilters
using POMCPOW
using Random
using Statistics

# using D3Trees
# using ProfileView

pomdp = LightDark1D()
ds0 = POMDPs.initialstate_distribution(pomdp)

up = BootstrapFilter(pomdp, 10000)

# function POMCPOW.state_weight(d::LDNormalStateDist, s::LightDark1DState)
#     dist = POMDPModels.Normal(d.mean, d.std)
#     return pdf(dist, s.y)
# end

function estimate_value(p::LightDark1D, s::LightDark1DState, h, steps)
    d = clamp(s.y - 1.0, 0.0, Inf)
    steps = ceil(d/p.step_size)
    POMDPs.discount(p)^steps # *10.0/2.0
    return 0.0
end

solver = POMCPOWSolver(tree_queries=500,
                        check_repeat_obs=true,
                        check_repeat_act=true,
                        estimate_value=estimate_value,
                        # next_action
                        k_action=10,
                        alpha_action=0.5,
                        k_observation=3,
                        alpha_observation=0.2,
                        rng=MersenneTwister(1)
                        )
planner = POMDPs.solve(solver, pomdp)

N = 1000
rs = RolloutSimulator(max_steps=500)
V = Float64[]
println("Starting simulations")
for i in 1:N
    if (i%25) == 0
        println("Trial $i")
    end
    s0 = rand(ds0)
    b0 = LDNormalStateDist(s0.y, 5.0)
    v = simulate(rs, pomdp, planner, up, b0, s0)
    push!(V, v)
end
mean_v = mean(V)
se_v = std(V)/sqrt(N)
println("Discounted Return: $mean_v ± $se_v")
