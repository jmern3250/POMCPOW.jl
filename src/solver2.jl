function simulate(pomcp::POMCPOWPlanner, h_node::POWTreeObsNode{B,A,O}, s::S, d) where {B,S,A,O}

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
                            false)
            end
        end
    end
    total_n = tree.total_n[h]

    best_node = select_best(pomcp.criterion, h_node, pomcp.solver.rng)
    a = tree.a_labels[best_node]

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
            push!(tree.total_n, 1)
            push!(tree.tried, Int[])
            push!(tree.o_labels, o)
			push!(tree.v, 0.0)

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
		if POMDPs.isterminal(pomcp.problem, sp) || d-1 <= 0
	        vp = 0.0
		else
			vp = estimate_value(pomcp.solved_estimate, pomcp.problem, sp, POWTreeObsNode(tree, hao), d-1)
		end
        # R = r + POMDPs.discount(pomcp.problem)*vp
		tree.v[hao] = vp
    else
        pair = rand(sol.rng, tree.generated[best_node])
        o = pair.first
        hao = pair.second
        push_weighted!(tree.sr_beliefs[hao], pomcp.node_sr_belief_updater, s, sp, r)
        sp, r = rand(sol.rng, tree.sr_beliefs[hao])
		if POMDPs.isterminal(pomcp.problem, sp) || d-1 <= 0
			tree.total_n[hao] += 1
		else
			simulate(pomcp, POWTreeObsNode(tree, hao), sp, d-1)
	    end
        # R = r + POMDPs.discount(pomcp.problem)*vp
    end

    tree.n[best_node] += 1
    tree.total_n[h] += 1

    # if tree.r[best_node] != -Inf
	tree.r[best_node] += (r-tree.r[best_node])/tree.n[best_node]
    # end
	child_a = unique([pair.second for pair in tree.generated[best_node]])
	q = 0.0
	for hao in child_a
		q += tree.v[hao]*tree.total_n[hao]
	end
	tree.q[best_node] = POMDPs.discount(pomcp.problem)*q/tree.n[best_node]
	tree.q[best_node] += tree.r[best_node]

	child_h = tree.tried[h]
	v = -Inf
	for ha in child_h
		# v += tree.q[ha]
		if tree.q[ha] > v
			v = tree.q[ha]
		end
	end
	tree.v[h] = v
    # return R
end
