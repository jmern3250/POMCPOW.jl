using Revise

using POMDPs
using POMDPSimulators
using POMCPOW
using Random
using Statistics

includet("../dev/pomdps/gaussian_pomdp.jl")

DIMS = 25
STEPS = 5

pomdp = GaussianPOMDP(DIMS, STEPS, 2, 3, 1.0) #dims, timesteps, delta, n_initial, l
ds0 = POMDPs.initialstate_distribution(pomdp)

up = GaussianProcessUpdater(DIMS, Float64, 0.0, 1.0)

solver = POMCPOWSolver(tree_queries=1000,
                        check_repeat_obs=true,
                        check_repeat_act=true,
                        # estimate_value=estimate_value,
                        # next_action
                        k_action=10,
                        alpha_action=0.5,
                        k_observation=3,
                        alpha_observation=0.2,
                        rng=MersenneTwister(1)
                        )
planner = POMDPs.solve(solver, pomdp)

N = 500
rs = RolloutSimulator(max_steps=500)
V = Float64[]
println("Starting simulations")
for i in 1:N
    if (i%25) == 0
        println("Trial $i")
    end
    s0 = rand(ds0)
    b0 = initialize_belief(up, s0)
    v = simulate(rs, pomdp, planner, up, b0, s0)
    push!(V, v)
end
mean_v = mean(V)
se_v = std(V)/sqrt(N)
println("Discounted Return: $mean_v ± $se_v")

# solver = POMCPOWSolver(max_depth=30, tree_queries=1000,
#                     criterion=MaxUCB(10.0),
#                     k_observation = 2.0,
#                     alpha_observation = 0.1,
#                     k_action = 3.,
#                     alpha_action = 0.25,
#                     # estimate_value=POMCPOW.RolloutEstimator(rollout_policy),
#                     )
#
# planner = POMDPs.solve(solver, pomdp)
# t1 = time_ns()
# r_total = 0.0
# for (s, bp, a, r, t) in stepthrough(pomdp, planner, belief_updater, b0, s0, "s,bp,a,r,t", max_steps=50)
#     # @show s
#     @show a
#     @show r
#     @show t
#     global t1
#     t2 = time_ns()
#     dt = (t2 - t1)/1.0e9
#     println("Elapsed: $dt (s)")
#     mu_sigma = GaussianProcessSolve(bp)
#     global DIMS
#     mu = reshape_gp_samples(mu_sigma[1], CartesianIndices((1:DIMS, 1:DIMS)), DIMS)
#     heatmap(mu,title="Belief Means Step $t", fill=true, clims=(-3,3), xlims = (0,DIMS), ylims = (0,DIMS))
#
#     branch_actions = planner.tree.a_labels[planner.tree.tried[1]]
#     n_acts = length(branch_actions)
#     true_acts = zeros(Float64, 2, 0)
#     false_acts = zeros(Float64, 2, 0)
#     for i = 1:n_acts
#         act_array = Float64[branch_actions[i][2][2] branch_actions[i][2][1]]
#         println(act_array)
#         act_array = reshape(act_array, 2, 1)
#         println(act_array)
#         if branch_actions[i][1]
#             true_acts = [true_acts act_array]
#         else
#             # append!(false_acts, act_array)
#             false_acts = [false_acts act_array]
#         end
#     end
#
#     println("HERE")
#     # true_acts = bp.x_act
#     # false_acts = bp.x_obs
#     fig = scatter!(false_acts[2,:], false_acts[1,:], legend=false, markershape=:hexagon)
#     fig = scatter!(true_acts[2,:], true_acts[1,:], legend=false, markershape=:square)
#     display(fig)
#
#     # true_acts = bp.x_act
#     # false_acts = bp.x_obs
#     # fig = scatter!(false_acts[2,:], false_acts[1,:], legend=false, markershape=:hexagon)
#     # fig = scatter!(true_acts[2,:], true_acts[1,:], legend=false, markershape=:square)
#     display(fig)
#     global r_total
#     r_total += POMDPs.discount(pomdp)^(t-1)*r
#
#     t1 = time_ns()
# end
# println("Total discounted rewards: $r_total")
#
# # action, info = action_info(planner, b0, tree_in_info=true)
# # tree = info[:tree]
# # inbrowser(D3Tree(info[:tree], init_expand=1), "firefox")
