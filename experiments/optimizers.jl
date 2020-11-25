struct AdamParams{T} <: Any where {T}
    α::T
    β1::T
    β2::T
    ϵ::T
    t::Array{Int,1}
    m::Array{T}
    v::Array{T}
    Δ::Array{T}

    function AdamParams(x, α::T; β1::T = 0.9, β2::T = 0.999, ϵ::T =1e-8) where {T}
        tp = eltype(x)
        m = zeros(tp, size(x))
        v = zeros(tp, size(x))
        Δ = zeros(tp, size(x))
        new{tp}(tp(α), β1, β2, ϵ, [0], m, v, Δ)
    end
end


function update!(opt::AdamParams, x, g)
    opt.t[1] += 1
    t, β1, β2, ϵ = opt.t[1], opt.β1, opt.β2, opt.ϵ
    @. opt.m *= β1
    @. opt.m += (1.0 - β1) * g
    @. opt.v *= β2
    @. opt.v += (1.0 - β2) * g^2
    α = opt.α * √(1.0 - β2^t) / (1.0 - β1^t)
    @. opt.Δ = α * opt.m / (√opt.v + ϵ)
    # add_to_params!(opt.f, @. α * opt.m / (√opt.v + ϵ))
    @. x += α * opt.m / (√opt.v + ϵ)
end

function optimize(params::AdamParams, g!, x₀, num_iters=100)
    G = similar(x₀)
    fill!(G, 0.0)
    x = deepcopy(x₀)
    for i in 1:num_iters
        g!(G, x)
        # println(i, " ", G)
        update!(params, x, G)
    end

    return x
end
