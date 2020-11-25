import EvaluationOfRLAlgs:AbstractPolicy, logprob, get_action!, gradient_logp!
import Base:length
using Random

function softmax(x)
    x = clamp.(x, -32., 32.)
    return exp.(x) / sum(exp.(x))
end

# function sample_discrete(p::Array{<:Real, 1}, rng)::Int
function sample_discrete(p, rng)::Int
    n = length(p)
    i = 1
    c = p[1]
    u = rand(rng)
    while c < u && i < n
        c += p[i += 1]
    end

    return i
end

mutable struct StatelessSoftmaxPolicy{TP} <: AbstractPolicy where {TP}
    θ::Array{TP}
    action::Int
    probs::Array{TP,1}
    function StatelessSoftmaxPolicy(::Type{T}, num_actions::Int) where {T}
        θ = zeros(T, num_actions)
        p = zeros(T, num_actions)
        new{T}(θ, 0, p)
    end
end

function logprob(π::StatelessSoftmaxPolicy, action)
    return log(π.probs[action])
end

function get_action_probabilities!(π::StatelessSoftmaxPolicy)
    π.probs .= softmax(π.θ)
end

function get_action!(π::StatelessSoftmaxPolicy, rng)
    # get_action_probabilities!(π)
    π.action = sample_discrete(π.probs, rng)
    logp = log(π.probs[π.action])
    return logp
end

function gradient_logp!(grad, π::StatelessSoftmaxPolicy, action::Int)
    fill!(grad, 0.)
    # get_action_probabilities!(π)
    logp = log(π.probs[action])

    grad .-= π.probs
    grad[action] += 1.

    return logp
end

function gradient_entropy!(grad, π::StatelessSoftmaxPolicy)
    # fill!(grad, 0.)

    # p1 = zeros(length(π.probs))
    @. grad = -π.probs * log(π.probs)
    H = sum(grad)  # entropy (p1 has negative in it)
    @. grad += π.probs * H
    return H
end

function gradient_entropy(π::StatelessSoftmaxPolicy)
    grad = similar(π.θ)
    gradient_entropy!(grad, π)
    return grad
end

function get_action_gradient_logp!(grad, π::StatelessSoftmaxPolicy, rng)
    logp = get_action!(π, rng)
    gradient_logp!(grad, π, π.action)
    return logp
end

function set_params!(π::StatelessSoftmaxPolicy{T}, θ) where {T}
    @. π.θ = θ
    get_action_probabilities!(π::StatelessSoftmaxPolicy)
end

function get_params(π::StatelessSoftmaxPolicy{T}) where {T}
    return π.θ
end

function copy_params!(params::Array{T}, π::StatelessSoftmaxPolicy{T}) where {T}
    @. params = π.θ
end

function copy_params(π::StatelessSoftmaxPolicy{T}) where {T}
    return deepcopy(π.θ)
end

function add_to_params!(π::StatelessSoftmaxPolicy{T}, Δθ) where {T}
    @. π.θ += Δθ
end

function clone(π::StatelessSoftmaxPolicy{T})::StatelessSoftmaxPolicy{T} where {T}
    π₂ = StatelessSoftmaxPolicy(T, length(π.θ))
    set_params!(π₂, π.θ)
    π₂.action = deepcopy(π.action)
    return π₂
end
