import EvaluationOfRLAlgs:AbstractPolicy, logprob, get_action!, gradient_logp!
import Base:length
using Random

function sample_normal!(z, μ, σ, rng)
    T = eltype(μ)
    @. z = μ + σ * randn((rng,), (T,))
end

function logpdf_normal(z, μ, σ)
    logp = -log(sqrt(2.0 * π)) - (z - μ)^2 / (2.0 * σ^2) - log(σ)
    return logp
end



mutable struct StatelessNormalPolicy{TP,TS} <: AbstractPolicy where {TP<:Real,TS<:Bool}
    σ::Array{TP,1}
    μ::Array{TP,1}
    action::Array{TP,1}

    function StatelessNormalPolicy(::Type{T}, num_actions::Int, sigma, train_sigma::Bool=true) where {T}
        σ = ones(T, num_actions) .* sigma
        a = zeros(T, num_actions)
        μ = zeros(T, num_actions)
        new{T,train_sigma}(σ, μ, a)
    end
end

function get_action!(π::StatelessNormalPolicy{T}, rng::AbstractRNG)::T where {T<:Real}
    sample_normal!(π.action, π.μ, π.σ, rng)
    logp = sum(logpdf_normal.(π.action, π.μ, π.σ))
    return logp
end

function get_action(π::StatelessNormalPolicy{T}, rng::AbstractRNG) where {T<:Real}
    a = π.μ .+ π.σ .* randn(rng, T, length(π.μ))
    logp = sum(logpdf_normal.(a, π.μ, π.σ))
    return a, logp
end


function gradient_logp!(grad::Array{T}, π::StatelessNormalPolicy{T,true}, action)::T where {T<:Real}
    fill!(grad, 0.)
    N = length(π.μ)
    @. grad[1:N] = (action - π.μ) / π.σ^2
    @. grad[N+1:end] = (-1 + ((action - π.μ) / π.σ)^2) / π.σ


    logp = sum(logpdf_normal.(action, π.μ, π.σ))

    return logp
end

function gradient_logp!(grad::Array{T}, π::StatelessNormalPolicy{T,false}, action)::T where {T<:Real}
    fill!(grad, 0.)
    @. grad = (action - π.μ) / π.σ^2
    logp = sum(logpdf_normal.(action, π.μ, π.σ))
    return logp
end

function gradlogpi!(g, θ, a)
    μ, σ = θ
    amu = a - μ
    @. g = amu / σ^2, (-1 + (amu / σ)^2) / σ
end

function logprob(π::StatelessNormalPolicy, action)
    return sum(logpdf_normal.(action, π.μ, π.σ))
end

function entropy(policy::StatelessNormalPolicy)
    return 0.5 * sum(log(2 * π * ℯ * policy.σ^2))  # note ℯ is \euler
end

function gradient_entropy(policy::StatelessNormalPolicy{TP,true}) where {TP}
    N = length(policy.μ)
    g = zeros(2*N)
    @. g[N+1:end] = (1.0 / policy.σ)
    return g
end

function gradient_entropy(policy::StatelessNormalPolicy{TP,false}) where {TP}
    return zeros(size(policy.μ))
end

function set_params!(π::StatelessNormalPolicy{T,true}, θ::Array{T}) where {T}
    N = length(π.μ)
    @. π.μ = θ[1:N]
    @. π.σ = θ[N+1:end]
    clamp!(π.σ, 0.001, 100)
end

function set_params!(π::StatelessNormalPolicy{T,false}, θ::Array{T}) where {T}
    @. π.μ = θ
end

function get_params(π::StatelessNormalPolicy{T,true}) where {T}
    return vcat(π.μ, π.σ)
end

function get_params(π::StatelessNormalPolicy{T,false}) where {T}
    return π.μ
end

function add_to_params!(π::StatelessNormalPolicy{T,true}, Δθ) where {T}
    N = length(π.μ)
    @. π.μ += Δθ[1:N]
    @. π.σ += θ[N+1:end]
    clamp!(π.σ, 0.001, 100)
end

function clone(π::StatelessNormalPolicy{T, TS})::StatelessNormalPolicy{T,TS} where {T,TS}
    A = length(π.σ)
    π₂ = StatelessNormalPolicy(T, A, π.σ, TS)
    π₂.action = deepcopy(π.action)
    π₂.μ = deepcopy(π.μ)
    return π₂
end
