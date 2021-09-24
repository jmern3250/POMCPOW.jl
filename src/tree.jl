struct POMCPOWTree{B,A,O,RB,S}
    # action nodes
    n::Vector{Int}
    v::Vector{Float64}
    w::Vector{Float64} # Sample weights for IS
    generated::Vector{Vector{Pair{O,Int}}}
    a_child_lookup::Dict{Tuple{Int,O}, Int} # may not be maintained based on solver params
    a_labels::Vector{A}
    n_a_children::Vector{Int}

    # observation nodes
    sr_beliefs::Vector{B} # first element is #undef
    total_n::Vector{Int}
    tried::Vector{Vector{Int}}
    o_child_lookup::Dict{Tuple{Int,A}, Int} # may not be maintained based on solver params
    o_labels::Vector{O}

    # root
    root_belief::RB
    root_samples::Vector{Tuple{S, UInt32}}
    root_action_samples::Vector{Vector{Tuple{S, UInt32}}}
    root_probs::Vector{Float64}
    root_returns::Vector{Float64}

    function POMCPOWTree{B,A,O,RB,S}(root_belief, sz::Int=1000) where{B,A,O,RB,S}
        sz = min(sz, 100_000)
        return new(
            sizehint!(Int[], sz),
            sizehint!(Int[], sz),
            sizehint!(Int[], sz),
            sizehint!(Vector{Pair{O,Int}}[], sz),
            Dict{Tuple{Int,O}, Int}(),
            sizehint!(A[], sz),
            sizehint!(Int[], sz),

            sizehint!(Array{B}(undef, 1), sz),
            sizehint!(Int[0], sz),
            sizehint!(Vector{Int}[Int[]], sz),
            Dict{Tuple{Int,A}, Int}(),
            sizehint!(Array{O}(undef, 1), sz),

            root_belief,
            sizehint!(Tuple{S, UInt32}[], sz),
            sizehint!(Vector{Tuple{S, UInt32}}[], sz),
            sizehint!(Float64[], sz),
            sizehint!(Float64[], sz)
        )
    end
end

@inline function push_anode!(tree::POMCPOWTree{B,A,O,RB,S}, h::Int, a::A, n::Int=0,
                            v::Float64=0.0, w::Float64=0.0, update_lookup=true) where {B,A,O,RB,S}
    anode = length(tree.n) + 1
    push!(tree.n, n)
    push!(tree.v, v)
    push!(tree.w, w)
    push!(tree.generated, Pair{O,Int}[])
    push!(tree.a_labels, a)
    push!(tree.n_a_children, 0)
    if update_lookup
        tree.o_child_lookup[(h, a)] = anode
    end
    push!(tree.tried[h], anode)
    push!(tree.root_action_samples, Tuple{S, UInt32}[])
    tree.total_n[h] += n
    return anode
end

struct POWTreeObsNode{B,A,O,RB} <: BeliefNode
    tree::POMCPOWTree{B,A,O,RB}
    node::Int
end

isroot(h::POWTreeObsNode) = h.node==1
@inline function belief(h::POWTreeObsNode)
    if isroot(h)
        return h.tree.root_belief
    else
        return StateBelief(h.tree.sr_beliefs[h.node])
    end
end
function sr_belief(h::POWTreeObsNode)
    if isroot(h)
        error("Tried to access the sr_belief for the root node in a POMCPOW tree")
    else
        return h.tree.sr_beliefs[h.node]
    end
end
n_children(h::POWTreeObsNode) = length(h.tree.tried[h.node])

function is_sample_root(tree::POMCPOWTree, a_idx::Int64)
    # μ = mean(tree.root_returns)
    root_probs = []
    root_returns = []
    root_samples = []
    action_samples = tree.root_action_samples[a_idx]
    for (idx, sample) in enumerate(tree.root_samples)
        if sample ∉ action_samples
            push!(root_probs, tree.root_probs[idx])
            push!(root_returns, tree.root_returns[idx])
            push!(root_samples, tree.root_samples[idx])
        end
    end
    p_total = sum(root_probs)
    p_norm = root_probs./p_total
    μ = sum(root_returns.*p_norm)
    wq = abs.(root_returns .- μ).*p_norm .+ 1e-4
    w_total = sum(wq)
    w_norm = wq./w_total
    wv = ProbabilityWeights(w_norm)
    idx = StatsBase.sample([1:length(root_samples);], wv)
    sample = root_samples[idx]
    wp = p_norm[idx]/w_norm[idx]
    return (idx, sample[1], sample[2], wp)
end
