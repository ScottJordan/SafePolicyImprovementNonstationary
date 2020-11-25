abstract type AbstractBandit end
abstract type AbstractDiscreteBandit <: AbstractBandit end
abstract type AbstractContinuousBandit <: AbstractBandit end

struct NonstationaryQuadraticBanditParams{T} <: AbstractContinuousBandit where {T}
    a::T
    b::T
    c::T
    τ::T
    t::Array{T,1}
end


struct DiscreteBanditParams{T} <: AbstractDiscreteBandit where {T}
    μ::Array{T,1}
    σ::Array{T,1}
end

struct NonStationaryDiscreteBanditParams{T} <: AbstractDiscreteBandit where {T}
    μ::Array{T,1}
    σ::Array{T,1}
    τ::Array{T,1}
    k::Array{T,1}
    t::Array{T,1}
end

function sample_reward!(b::DiscreteBanditParams{T}, action::Int, rng) where {T}
    return b.μ[action] + b.σ[action] * randn(rng)
end

function sample_reward!(b::NonStationaryDiscreteBanditParams{T}, action::Int, rng) where {T}
    r = b.μ[action] * sin(b.t[1] * b.τ[action] + b.k[action]) + b.σ[action] * randn(rng)
    # println("$action $(b.t[1]) $(b.τ[action]) $(b.k[action]) ", b.t[1] * b.τ[action] + b.k[action], " ", sin(b.t[1] * b.τ[action] + b.k[action]))
    b.t[1] += 1.0
    return r
end

function sample_reward!(b::NonstationaryQuadraticBanditParams{T}, action, rng) where {T}
    r = -0.1*(b.a*action[1] + b.b*cos(b.t[1] * b.τ))^2 + b.c * randn(rng)
    b.t[1] += 1.0
    return r
end

function eval_policy(b::NonstationaryQuadraticBanditParams{T}, π::StatelessNormalPolicy) where {T,T2}
    t = b.t[1]
    μ, σ = π.μ[1], π.σ[1]
    J = -0.1*b.a^2^σ^2  + -0.1*(b.a*μ + b.b*cos(t * b.τ))^2
    return J
end

function eval_policy(b::T, π::StatelessSoftmaxPolicy) where {T <: AbstractDiscreteBandit}
    return eval_policy(b, π.probs)
end

function eval_policy(b::NonStationaryDiscreteBanditParams{T}, p::Array{T2}) where {T,T2}
    J = 0.0
    t = b.t[1]
    for i in 1:length(p)
        J += p[i] * b.μ[i] * sin(t * b.τ[i] + b.k[i])
    end
    return J
end

function eval_policy(b::DiscreteBanditParams{T}, p::Array{T2}) where {T,T2}
    J = 0.0
    t = b.t[1]
    for i in 1:length(p)
        J += p[i] * b.μ[i]
    end
    return J
end
