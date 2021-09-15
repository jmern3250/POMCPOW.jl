using Revise

using POMDPs
using POMDPModels
using POMDPModels: LDNormalStateDist, LightDark1DState
using POMDPSimulators
using ParticleFilters
using POMCPOW

m = LightDark1D()
ds0 = POMDPs.initialstate_distribution(m)
s0 = rand(ds0)

up = BootstrapFilter(m, 1000);
pomdp = LightDark1D()
N=1000

up = BootstrapFilter(pomdp, N)
b0 = LDNormalStateDist(s0.y, 5.0)

function POMCPOW.state_weight(d::LDNormalStateDist, s::LightDark1DState)
    dist = POMDPModels.Normal(d.mean, d.std)
    return pdf(dist, s.y)
end

solver = POMCPOWSolver(tree_queries=10,
                        check_repeat_obs=false,
                        check_repeat_act=true,
                        k_observation=2,
                        alpha_observation=0.1
                        )
planner = POMDPs.solve(solver, m)

a, info = POMCPOW.action_info(planner, b0, tree_in_info=true)
tree = info[:tree]
# println("Entering Simulation...")
# for (s, a, r, bp, t) in stepthrough(m, planner, up, b0, s0, "s,a,r,bp,t", max_steps=50)
#     @show a
#     @show r
#     @show t
# end
