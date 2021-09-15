using Revise

using POMDPs
using POMDPModels
using POMDPModels: LDNormalStateDist, LightDark1DState
using POMDPSimulators
using ParticleFilters
using POMCPOW
# using MCVI
using Statistics

using ProfileView

pomdp = LightDark1D()
ds0 = POMDPs.initialstate_distribution(pomdp)
s0 = rand(ds0)

up = BootstrapFilter(pomdp, 1000)
b0 = LDNormalStateDist(s0.y, 5.0)

function POMCPOW.state_weight(d::LDNormalStateDist, s::LightDark1DState)
    dist = POMDPModels.Normal(d.mean, d.std)
    return pdf(dist, s.y)
end

solver = POMCPOWSolver(tree_queries=1000,
                        max_samples=100,
                        check_repeat_obs=true,
                        check_repeat_act=true,
                        k_observation=2,
                        alpha_observation=0.1,
                        estimate_value,
                        next_action
                        )
planner = POMDPs.solve(solver, pomdp)

# mcvi = MCVISolver(10, 100, 8, 500, 1000, 5000, 50)
# policy = solve(mcvi, pomdp)

# a, info =
# @profview POMCPOW.action_info(planner, b0, tree_in_info=true)
# @profview POMCPOW.action_info(planner, b0, tree_in_info=true)
# tree = info[:tree]

# V = 0.0
# println("Entering Simulation...")
# for (s, a, r, bp, t) in stepthrough(m, planner, up, b0, s0, "s,a,r,bp,t", max_steps=50)
#     global V
#     @show a
#     @show r
#     @show t
#     # @show
#     # println(typeof(bp))
#     V += POMDPs.discount(pomdp)^(t-1)*r
# end

N = 50
rs = RolloutSimulator()
V = Float64[]
println("Starting simulations")
for i in 1:N
    println("Trial $i")
    v = simulate(rs, pomdp, planner)
    push!(V, v)
end
mean_v = mean(V)
se_v = std(V)/sqrt(N)
println("Discounted Return: $mean_v ± $se_v")
