using EvaluationOfRLAlgs

abstract type AbstractImportanceSampling end
abstract type UnweightedIS <: AbstractImportanceSampling end
abstract type WeightedIS <: AbstractImportanceSampling end

struct ImportanceSampling <: UnweightedIS end
struct PerDecisionImportanceSampling <: UnweightedIS end
struct WeightedImportanceSampling <: WeightedIS end
struct WeightedPerDecisionImportanceSampling <: WeightedIS end

function IS_weight(blogp, a, π)
    return exp(logprob(π, a) - blogp)
end

function IS_weight(blogp, logp)
    return exp(logp - blogp)
end

function estimate_return!(G, H::BanditHistory, π, ::UnweightedIS)
    @. G = IS_weight(H.blogps, H.actions, (π,)) * H.rewards
end

function estimate_entropyreturn!(G, H::BanditHistory, π, α, ::UnweightedIS)
    # @. G = IS_weight(H.blogps, H.actions, (π,)) * H.rewards
    for t in 1:length(G)
        logp = logprob(π, H.actions[t])
        G[t] = IS_weight(H.blogps[t], logp) * (H.rewards[t] - α * logp)
    end
end

function estimate_entropyreturn!(G, H::BanditHistory, idxs, π, α, ::UnweightedIS)
    # @. G = IS_weight(H.blogps, H.actions, (π,)) * H.rewards
    for (i,t) in enumerate(idxs)
        logp = logprob(π, H.actions[t])
        G[i] = IS_weight(H.blogps[t], logp) * (H.rewards[t] - α * logp)
    end
end

function estimate_return!(G, H::BanditHistory, idxs, π, ::UnweightedIS)
    blogps = view(H.blogps, idxs)
    actions = view(H.actions, idxs)
    rewards = view(H.rewards, idxs)
    @. G = IS_weight(blogps, actions, (π,)) * rewards
end

function estimate_return!(G, H::BanditHistory, π, ::WeightedIS)
    W = @. IS_weight(H.blogps, H.actions, (π,))
    W ./= sum(W)
    @. G = W * H.rewards
end

function estimate_return!(G, H::History{T}, π, ::UnweightedIS) where {T<:Trajectory}
    println("error! estimating return not implemented for History of Trajectories")
end
