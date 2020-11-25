import Base:length,push!


struct BanditHistory{T,TA} <: Any where {T,TA}
    actions::Array{TA,1}
    blogps::Array{T,1}
    rewards::Array{T,1}

    function BanditHistory(::Type{T}, ::Type{TA}) where {T,TA}
        new{T,TA}(Array{TA,1}(), Array{T,1}(), Array{T,1}())
    end
end

struct Trajectory{T,TS,TA} <: Any where {T,TS,TA}
    states::Array{TS,1}
    actions::Array{TA,1}
    blogps::Array{T,1}
    rewards::Array{T,1}

    function Trajectory(::Type{T}, ::Type{TS}, ::Type{TA}) where {T,TS,TA}
        new{T,TS,TA}(Array{TS,1}(), Array{TA,1}(), Array{T,1}(), Array{T,1}())
    end
end

struct History{T} <: Any where {T}
    τs::Array{T,1}

    function History(::Type{T}) where {T,TA}
        new{T}(Array{T,1}())
    end
end

function length(H::BanditHistory)
    return length(H.rewards)
end

function length(H::History)
    return length(H.τs)
end

function push!(H::History{T}, item::T) where {T}
    push!(H.τs, item)
end

function push!(H::BanditHistory{T,TA}, action::TA, blogp::T, reward::T) where {T,TA}
    push!(H.actions, action)
    push!(H.blogps, blogp)
    push!(H.rewards, reward)
end

function collect_data!(H::BanditHistory, π, sample_fn!, N, rng)
    for i in 1:N
        logp = get_action!(π, rng)
        r = sample_fn!(π.action, rng)
        push!(H, deepcopy(π.action), logp, r)
    end
end
