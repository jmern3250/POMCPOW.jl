function simulate(pomcp::POMCPOWPlanner, h_node::POWTreeObsNode{B,A,O}, s::Union{Nothing, S},
                    w::Float64, d) where {B,S,A,O}

    if (s!= nothing && POMDPs.isterminal(pomcp.problem, s)) || d <= 0
        return 0.0, nothing
    end
    tree = h_node.tree
    h = h_node.node
    sol = pomcp.solver

    if sol.enable_action_pw
        total_n = tree.total_n[h]
        if length(tree.tried[h]) <= sol.k_action*total_n^sol.alpha_action
            if h == 1
                a = next_action(pomcp.next_action, pomcp.problem, tree.root_belief, POWTreeObsNode(tree, h))
            else
                a = next_action(pomcp.next_action, pomcp.problem, StateBelief(tree.sr_beliefs[h]), POWTreeObsNode(tree, h))
            end
            if !sol.check_repeat_act || !haskey(tree.o_child_lookup, (h,a))
                push_anode!(tree, h, a,
                            init_N(pomcp.init_N, pomcp.problem, POWTreeObsNode(tree, h), a),
                            init_V(pomcp.init_V, pomcp.problem, POWTreeObsNode(tree, h), a),
                            w,
                            sol.check_repeat_act)
            end
        end
    else # run through all the actions
        if isempty(tree.tried[h])
            if h == 1
                action_space_iter = POMDPs.actions(pomcp.problem, tree.root_belief)
            else
                action_space_iter = POMDPs.actions(pomcp.problem, StateBelief(tree.sr_beliefs[h]))
            end
            anode = length(tree.n)
            for a in action_space_iter
                push_anode!(tree, h, a,
                            init_N(pomcp.init_N, pomcp.problem, POWTreeObsNode(tree, h), a),
                            init_V(pomcp.init_V, pomcp.problem, POWTreeObsNode(tree, h), a),
                            w,
                            false)
            end
        end
    end
    total_n = tree.total_n[h]

    best_node = select_best(pomcp.criterion, h_node, pomcp.solver.rng)
    a = tree.a_labels[best_node]

    idx = nothing
    if s == nothing
        # if true
        if length(tree.root_samples) <= tree.n[best_node]
            s = rand(pomcp.solver.rng, tree.root_belief)
            seed = rand(UInt32)
            sw = state_weight(tree.root_belief, s)
            push!(tree.root_samples, (s, seed))
            push!(tree.root_probs, sw)
            push!(tree.root_returns, -Inf)
            wp = 1.0
            idx = length(tree.root_samples)
        else
            idx, s, seed, wp = is_sample_root(tree)
        end
        Random.seed!(pomcp.solver.rng, seed)
    end

    if POMDPs.isterminal(pomcp.problem, s)
        return 0.0, nothing
    end

    new_node = false
    if tree.n_a_children[best_node] <= sol.k_observation*(tree.n[best_node]^sol.alpha_observation)

        sp, o, r = @gen(:sp, :o, :r)(pomcp.problem, s, a, sol.rng)

        if sol.check_repeat_obs && haskey(tree.a_child_lookup, (best_node,o))
            hao = tree.a_child_lookup[(best_node, o)]
        else
            new_node = true
            hao = length(tree.sr_beliefs) + 1
            push!(tree.sr_beliefs,
                  init_node_sr_belief(pomcp.node_sr_belief_updater,
                                      pomcp.problem, s, a, sp, o, r))
            push!(tree.total_n, 0)
            push!(tree.tried, Int[])
            push!(tree.o_labels, o)

            if sol.check_repeat_obs
                tree.a_child_lookup[(best_node, o)] = hao
            end
            tree.n_a_children[best_node] += 1
        end
        push!(tree.generated[best_node], o=>hao)
    else

        sp, r = @gen(:sp, :r)(pomcp.problem, s, a, sol.rng)

    end

    if r == Inf
        @warn("POMCPOW: +Inf reward. This is not recommended and may cause future errors.")
    end

    if new_node
        R = r + POMDPs.discount(pomcp.problem)*estimate_value(pomcp.solved_estimate, pomcp.problem, sp, POWTreeObsNode(tree, hao), d-1)
    else
        pair = rand(sol.rng, tree.generated[best_node])
        o = pair.first
        hao = pair.second
        push_weighted!(tree.sr_beliefs[hao], pomcp.node_sr_belief_updater, s, sp, r)
        sp, r = rand(sol.rng, tree.sr_beliefs[hao])

        R = r + POMDPs.discount(pomcp.problem)*simulate(pomcp, POWTreeObsNode(tree, hao), sp, w, d-1)[1]
    end

    tree.n[best_node] += 1
    tree.total_n[h] += 1
    if tree.v[best_node] != -Inf
        tree.v[best_node] += (R-tree.v[best_node])/tree.n[best_node]
        # tree.w[best_node] += w
        # tree.v[best_node] += w*(R-tree.v[best_node])/tree.w[best_node]
    end

    return R, idx
end
