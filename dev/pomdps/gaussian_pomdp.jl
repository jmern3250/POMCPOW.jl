using POMDPs
using Random
using GaussianProcesses
using POMDPModelTools

function cartind_to_array(X)
    flattened = X[:]
    out = zeros(Float64, 2, 0)
    for i=1:length(flattened)
        x = reshape(Float64[flattened[i][1], flattened[i][2]], 2, 1)
        out = [out x]
    end
    return out
end

function reshape_gp_samples(X, idxs, dim)
    flattened_idxs = idxs[:]
    Y = zeros(Float64, dim, dim)
    for i=1:length(flattened_idxs)
        idx = flattened_idxs[i]
        y = X[i]
        Y[idx] = y
    end
    return Y
end

struct GaussianProcessState
    x_obs::Array{Float64, 2}
    x_act::Array{Float64, 2}
    y::Array{Float64, 1}
    full::Union{Array{Float64}, Nothing}
end

struct GaussianProcessBelief
    dim::Int #Grid size (e.g. 25 = gridworld 25x25)
    x_obs::Array
    x_act::Union{Nothing, Array}
    y::Array
    m::AbstractFloat
    l::AbstractFloat
    gp::GPE
    function GaussianProcessBelief(dim::Int, x_obs::Array, x_act::Union{Nothing, Array}, y::Array, m::Real=0, l::AbstractFloat=0.)
        gp = GPE(x_obs, y, MeanConst(m), SEIso(l,0.), -3.0)
        new(dim, x_obs, x_act, y, m, l, gp)
    end
end

function GaussianProcessBelief(dim::Int, dtype::Type=Float64, m::Float64=0.0, l::AbstractFloat=0.0)
    x_obs = Array{dtype}(undef, 2, 0)
    y = Array{dtype}(undef, 0)
    GaussianProcessBelief(dim, x_obs, nothing, y, m, l)
end

struct GaussianProcessUpdater <: POMDPs.Updater
    dim::Int
    dtype::Type
    m::AbstractFloat
    l::AbstractFloat
end

function POMDPs.update(b::GaussianProcessUpdater, old_b, action, obs)
    act = action[1]
    location = action[2]
    act_array = reshape(Float64[location[1] location[2]], 2, 1)
    x_obs = [old_b.x_obs act_array] # This is specific to our problem formulation where action:=location
    y = [old_b.y; obs]
    if act
        x_act = [old_b.x_act act_array]
    else
        x_act = old_b.x_act
    end
    GaussianProcessBelief(old_b.dim, x_obs, x_act, y, old_b.m, old_b.l)
end

function POMDPs.initialize_belief(up::GaussianProcessUpdater, s::GaussianProcessState)
    GaussianProcessBelief(up.dim, s.x_obs, s.x_act, s.y, up.m, up.l)
end

function POMDPs.initialize_belief(up::GaussianProcessUpdater, b::GaussianProcessBelief)
    GaussianProcessBelief(up.dim, b.x_obs, b.x_act, b.y, up.m, up.l)
end

function GaussianProcessSolve(b::GaussianProcessBelief; cov_matrix::Bool=false)
    x_out = Array{Float64}(undef, b.dim, b.dim)
    x_idxs = CartesianIndices((1:b.dim, 1:b.dim))
    x_grid = cartind_to_array(x_idxs)
    predict_y(b.gp, x_grid, full_cov=cov_matrix)
end

function GaussianProcessSolve(b::GaussianProcessBelief, x_grid::Array; cov_matrix::Bool=false)
    predict_y(b.gp, x_grid, full_cov=cov_matrix)
end

function GaussianProcessSample(b::GaussianProcessBelief)
    x_out = Array{Float64}(undef, b.dim, b.dim)
    x_idxs = CartesianIndices((1:b.dim, 1:b.dim))
    x_grid = cartind_to_array(x_idxs)
    GaussianProcesses.rand(b.gp, x_grid)
end

function POMDPs.rand(rng::AbstractRNG, b::GaussianProcessBelief)
    GaussianProcessState(b.x_obs, b.x_act, b.y, nothing)
end

struct GaussianPOMDP <: POMDP{GaussianProcessState, Tuple{Bool,CartesianIndex}, Float64}
    dim::Int
    timesteps::Int
    delta::Int # Minimum distance between actions (in x and y)
    n_initial::Int # Number of initial sensor positions
    l::Float64 # Length Scale for generative Gaussian Process
    function GaussianPOMDP(gridDim::Int, timesteps::Int, delta::Int, n_initial::Int, l::Float64)
        # May use in future to enforce constraints on dimensions vs no initial samples
        new(gridDim, timesteps, delta, n_initial, l)
    end
end

function POMDPs.gen(m::GaussianPOMDP, s, a, rng)
    # transition model
    x_obs = s.x_obs
    x_act = s.x_act
    y_obs = s.y
    full = s.full
    act = a[1]
    loc = a[2]
    if isnothing(full)
        gp = GPE(x_obs, y_obs, MeanZero(), SEIso(m.l, 0.0), -3.)
        a = reshape(Float64[loc[1] loc[2]], 2, 1)
        o = Base.rand(gp, a)[1]
    else
        o = full[loc]
        a = reshape(Float64[loc[1] loc[2]], 2, 1)
    end
    o = clamp(o, -3, 3)
    o = round(o/0.1, digits=0)*0.1
    if act
        r = o #+ 3.0
        xp_act = [x_act a]
    else
        xp_act = x_act
        # r = -0.1
        r = 0
    end
    xp_obs = [x_obs a]
    yp = [y_obs; o]
    sp = GaussianProcessState(xp_obs, xp_act, yp, full)
    return (sp=sp, o=o, r=r)
end

function POMDPs.reward(p::GaussianPOMDP, s, a, sp)
    act = a[1]
    if act
        return sp.y[end]
    else
        return 0
    end
end

struct GaussianInitialStateDistribution
    dim::Int
    n_initial::Int
    l::Float64
    gp::GPE
    function GaussianInitialStateDistribution(dim::Int, n_initial::Int, l::Float64)
        x = Array{Float64}(undef, 2, 0)
        y = Array{Float64}(undef, 0)
        gp = GPE(x, y, MeanZero(), SEIso(l, 0.0), -3.)
        new(dim, n_initial, l, gp)
    end
end

POMDPs.initialstate_distribution(m::GaussianPOMDP) = GaussianInitialStateDistribution(m.dim, m.n_initial, m.l)

function Base.rand(d::GaussianInitialStateDistribution)
    n = d.n_initial
    idxs = CartesianIndices((1:d.dim, 1:d.dim))
    idx_array = cartind_to_array(idxs)
    full = Base.rand(d.gp, idx_array)
    full = clamp.(full, -3, 3)
    full = round.(full./0.1, digits=0)*0.1
    full = reshape_gp_samples(full, idxs, d.dim)
    x_idxs = rand(idxs[:], d.n_initial)
    x = cartind_to_array(x_idxs)
    y = full[x_idxs]
    x_act = zeros(Float64, 2, 0)
    s = GaussianProcessState(x, x_act, y, full)
end

function Base.rand(rng::AbstractRNG, d::GaussianInitialStateDistribution)
    n = d.n_initial
    idxs = CartesianIndices((1:d.dim, 1:d.dim))
    idx_array = cartind_to_array(idxs)
    full = Base.rand(d.gp, idx_array)
    full = clamp.(full, -3, 3)
    full = round.(full./0.1, digits=0)*0.1
    full = reshape_gp_samples(full, idxs, d.dim)
    x_idxs = rand(idxs[:], d.n_initial)
    x = cartind_to_array(x_idxs)
    y = full[x_idxs]
    x_act = zeros(Float64, 2, 0)
    s = GaussianProcessState(x, x_act, y, full)
end

function POMDPs.actions(p::GaussianPOMDP)
    locs = CartesianIndices((1:p.dim, 1:p.dim))[:]
    acts = []
    for i=1:length(locs)
        loc = locs[i]
        push!(acts, (true, loc))
        push!(acts, (false, loc))
    end
    return acts
end

function POMDPs.actions(p::GaussianPOMDP, s::GaussianProcessState)
    actions = Set(POMDPs.actions(p))
    for i = 1:size(s.x_act)[2]
        loc = convert(Array{Int}, s.x_act[:, i])
        idxs = CartesianIndices((loc[1]-p.delta:loc[1]+p.delta, loc[2]-p.delta:loc[2]+p.delta))[:]
        true_set = Set(collect(((true, item) for item in idxs)))
        false_set = Set(collect(((false, item) for item in idxs)))
        diff_set = union(true_set, false_set)
        setdiff!(actions, diff_set)
    end
    collect(actions)
end

function POMDPs.actions(p::GaussianPOMDP, b::GaussianProcessBelief)
    actions = Set(POMDPs.actions(p))
    for i = 1:size(b.x_act)[2]
        loc = convert(Array{Int}, b.x_act[:, i])
        idxs = CartesianIndices((loc[1]-p.delta:loc[1]+p.delta, loc[2]-p.delta:loc[2]+p.delta))[:]
        true_set = Set(collect(((true, item) for item in idxs)))
        false_set = Set(collect(((false, item) for item in idxs)))
        diff_set = union(true_set, false_set)
        # setdiff!(actions, diff_set)
        setdiff!(actions, true_set)
    end
    collect(actions)
end

POMDPs.discount(::GaussianPOMDP) = 0.95
POMDPs.isterminal(p::GaussianPOMDP, s::GaussianProcessState) = (size(s.x_act, 2) >= p.timesteps || size(s.x_obs, 2) >= 20)

function plot_actions(acts)
    true_acts = zeros(Float64, 2, 0)
    false_acts = zeros(Float64, 2, 0)
    for act in acts
        act_array = Float64[act[2][1]; act[2][2]]
        if act[1]
            true_acts = [true_acts  act_array]
        else
            false_acts = [false_acts act_array]
        end
    end
    return (true_acts, false_acts)
end

function find_obs(s::GaussianProcessState, x::Array{Float64})
    idx = nothing
    for i=1:size(s.x_obs)[2]
        x_obs = s.x_obs[:,i]
        if x_obs == x
            idx = i
            break
        end
    end
    return s.y[idx]
end

function POMDPModelTools.obs_weight(p::GaussianPOMDP, s::GaussianProcessState, a::Tuple{Bool,CartesianIndex{2}}, sp::GaussianProcessState, o::Float64)
    act, loc = a
    a_array = Float64[loc[1], loc[2]]
    obs_s = find_obs(sp, a_array)
    if obs_s == o
        return 1.
    else
        return 0.
    end
end
