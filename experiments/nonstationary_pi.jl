using Zygote
using Statistics

include("history.jl")
include("offpolicy.jl")
include("highconfidence.jl")
include("nonstationary_modeling.jl")

function nswildbst_CI(π, δ, tail, D, idxs, ϕ, τ, num_boot, IS, rng)
    L = length(D)

    Φ, ϕτ = create_features(ϕ, idxs, collect(Float64, L+1:L+τ))
    A, B, C = get_coefst(Φ, ϕτ)

    N = length(idxs)
    Y = zeros(Float64, N)
    estimate_return!(Y, D, idxs, π, IS)
    if tail == :left
        return wildbst_CI(get_preds_and_residual_t(Y, A, B, C)..., B, C, δ, num_boot, rng)
    elseif tail == :right
        return wildbst_CI(get_preds_and_residual_t(Y, A, B, C)..., B, C, 1-δ, num_boot, rng)
    elseif tail == :both
        return wildbst_CI(get_preds_and_residual_t(Y, A, B, C)..., B, C, [δ/2.0, 1-(δ/2.0)], num_boot, rng)
    else
        println("ERROR tail: '$tail' not recognized. Returning NaN")
        return NaN
    end
    return
end

function nsbst_lower_grad(θ₀, D::BanditHistory{T,TA}, idxs, π, ϕ, τ, num_boot, δ, λ, IS::TI, rng) where {T,TA,TI<:UnweightedIS}
    L = length(D)
    N = length(idxs)
    Y = zeros(T, N)
    GY = similar(Y)
    ψ = similar(θ₀)

    Φ, ϕτ = create_features(ϕ, idxs, collect(Float64, L+1:L+τ))
    A, B, C = get_coefst(Φ, ϕτ)

    function g!(G, θ, Y, GY, ψ, π, D, idxs, A, B, C, δ, num_boot, IS, rng)
        set_params!(π, θ)
        estimate_return!(Y, D, idxs, π, IS)
        GY .= Zygote.gradient(y->wildbst_CI(get_preds_and_residual_t(y, A, B, C)..., B, C, δ, num_boot, rng), Y)[1]
        G .= λ .* gradient_entropy(π)

        for (i,idx) in enumerate(idxs)
            gradient_logp!(ψ, π, D.actions[idx])
            @. G += ψ * Y[i] * GY[i]
        end
        G ./= length(idxs)
    end

    return (G, θ)->g!(G, θ, Y, GY, ψ, π, D, idxs, A, B, C, δ, num_boot, IS, rng)
end


function nswildbs_CI(π, δ, tail, D, idxs, ϕ, τ, num_boot, aggf, IS, rng)
    L = length(D)

    Φ, ϕτ = create_features(ϕ, idxs, collect(Float64, L+1:L+τ))
    A, B = get_coefs(Φ, ϕτ)

    N = length(idxs)
    Y = zeros(Float64, N)
    estimate_return!(Y, D, idxs, π, IS)
    if tail == :left
        return wildbs_CI(get_preds_and_residual(Y, A, B)..., B, δ, num_boot, aggf, rng)
    elseif tail == :right
        return wildbs_CI(get_preds_and_residual(Y, A, B)..., B, 1-δ, num_boot, aggf, rng)
    elseif tail == :both
        return wildbs_CI(get_preds_and_residual(Y, A, B)..., B, [δ/2.0, 1-(δ/2.0)], num_boot, aggf, rng)
    else
        println("ERROR tail: '$tail' not recognized. Returning NaN")
        return NaN
    end
end

function off_policy_natgrad_bs!(G, θ, Y, GY, ψ, F, π, D::BanditHistory{T,TA}, idxs, A, B, δ, num_boot, aggf, λ, IS::TI, rng) where {T,TA,TI<:UnweightedIS}
    set_params!(π, θ)
    compute_fisher!(F, ψ, π)
    
    estimate_entropyreturn!(Y, D, idxs, π, λ, IS)
    GY .= Zygote.gradient(y->wildbs_CI(get_preds_and_residual(y, A, B)..., B, δ, num_boot, aggf, rng), Y)[1]
    
    fill!(G, 0.0)

    for (i,idx) in enumerate(idxs)
        gradient_logp!(ψ, π, D.actions[idx])
        @. G += ψ * Y[i] * GY[i]
    end
    G ./= length(idxs)
    G .= inv(F) * G
end

function off_policy_natgrad_bs!(G, θ, Y, GY, ψ, F, π::StatelessNormalPolicy, D::BanditHistory{T,TA}, idxs, A, B, δ, num_boot, aggf, λ, IS::TI, rng) where {T,TA,TI<:UnweightedIS}
    set_params!(π, θ)
    compute_fisher!(F, ψ, π)

    estimate_entropyreturn!(Y, D, idxs, π, λ, IS)
    GY .= Zygote.gradient(y->wildbs_CI(get_preds_and_residual(y, A, B)..., B, δ, num_boot, aggf, rng), Y)[1]

    fill!(G, 0.0)
    for (i,idx) in enumerate(idxs)
        logp = gradient_logp!(ψ, π, D.actions[idx])
        @. G += ψ * Y[i] * GY[i]
    end
    G ./= length(idxs)
    G .= inv(F) * G
end

function off_policy_natgrad_bs_old!(G, θ, Y, GY, ψ, F, π::StatelessNormalPolicy, D::BanditHistory{T,TA}, idxs, A, B, δ, num_boot, aggf, λ, IS::TI, rng) where {T,TA,TI<:UnweightedIS}
    set_params!(π, θ)
    compute_fisher!(F, ψ, π)

    estimate_return!(Y, D, idxs, π, IS)
    GY .= Zygote.gradient(y->wildbs_CI(get_preds_and_residual(y, A, B)..., B, δ, num_boot, aggf, rng), Y)[1]

    fill!(G, 0.0)
    G .= λ .* gradient_entropy(π)
    for (i,idx) in enumerate(idxs)
        logp = gradient_logp!(ψ, π, D.actions[idx])
        @. G += ψ * Y[i] * GY[i]
    end

    G ./= length(idxs)
    G .= inv(F) * G
end

function nsbs_lower_grad(θ₀, D::BanditHistory{T,TA}, idxs, π, ϕ, τ, num_boot, δ, aggf, λ, IS::TI, old_ent, rng) where {T,TA,TI<:UnweightedIS}
    L = length(D)
    N = length(idxs)
    Y = zeros(T, N)
    GY = similar(Y)
    ψ = similar(θ₀)

    Φ, ϕτ = create_features(ϕ, idxs, collect(Float64, L+1:L+τ))
    A, B = get_coefs(Φ, ϕτ)

    F = zeros(T, (length(ψ), length(ψ)))

    if old_ent
        return (G, θ)->off_policy_natgrad_bs_old!(G, θ, Y, GY, ψ, F, π, D, idxs, A, B, δ, num_boot, aggf, λ, IS, rng)
    else
        return (G, θ)->off_policy_natgrad_bs!(G, θ, Y, GY, ψ, F, π, D, idxs, A, B, δ, num_boot, aggf, λ, IS, rng)
    end
end

function maximum_entropy_fit(D::BanditHistory, idxs, π::StatelessNormalPolicy)
    adist = hcat(D.actions...)
    μ = mean(adist, dims=2)[:, 1]
    σ = std(adist, dims=2)[:, 1]
    return vcat(μ, σ)
end

function compute_fisher!(F, ψ, π::StatelessSoftmaxPolicy)
    fill!(F, 0.0)
    F .= F + I*1e-4
    for a in 1:length(π.probs)
        gradient_logp!(ψ, π, a)
        F .+= π.probs[a] .* (ψ * ψ')
    end
end

function compute_fisher!(F, ψ, π::StatelessNormalPolicy)
    fill!(F, 0.0)
    N = length(π.σ)
    F[diagind(F)] .= vcat(π.σ..., 2.0*π.σ...)
end

function maximize_nsbs_lower!(params, π, D, idxs, ϕ, τ, num_boot, δ, aggf, λ, IS, old_ent, num_iters, rng)
    θ = get_params(π)
    g! = nsbs_lower_grad(θ, D, idxs, π, ϕ, τ, num_boot, δ, aggf, λ, IS, old_ent, rng)
    result = optimize(params, g!, θ, num_iters)
    @. θ = result
    set_params!(π, θ)
end


function maximize_nsbst_lower!(params, π, D, idxs, ϕ, τ, num_boot, δ, λ, IS, old_ent, num_iters, rng)
    θ = get_params(π)
    g! = nsbst_lower_grad(θ, D, idxs, π, ϕ, τ, num_boot, δ, λ, IS, old_ent, rng)
    result = optimize(params, g!, θ, num_iters)
    @. θ = result
    set_params!(π, θ)
end


function maximize_ns_lower!(params, π, D, idxs, ϕ, τ, num_boot, δ, λ, IS, old_ent, num_iters, rng)
    θ = get_params(π)
    g! = ns_lower_grad(θ, D, idxs, π, ϕ, τ, num_boot, δ, λ, IS, old_ent, rng)
    result = optimize(params, g!, θ, num_iters)
    @. θ = result
    set_params!(π, θ)
end




function build_nsbs(D, π, oparams, ϕ, τ; nboot_train=200, nboot_test=500, δ=0.05, aggf=mean, λ=0.01, IS=PerDecisionImportanceSampling(), old_ent=false, num_iters=100, rng=Base.GLOBAL_RNG)
    opt_fun(oparams, π, D, idxs) = maximize_nsbs_lower!(oparams, π, D, idxs, ϕ, τ, nboot_train, δ, aggf, λ, IS, old_ent, num_iters, rng)
    bound_fun(D, idxs, π, δ, tail) = nswildbs_CI(π, δ, tail, D, idxs, ϕ, τ, nboot_test, aggf, IS, rng)
    return opt_fun, bound_fun
end

function build_nsbst(D, π, oparams, ϕ, τ; nboot_train=200, nboot_test=500, δ=0.05, λ=0.01, IS=PerDecisionImportanceSampling(), old_ent=false, num_iters=100, rng=Base.GLOBAL_RNG)
    opt_fun(oparams, π, D, idxs) = maximize_nsbs_lower!(oparams, π, D, idxs, ϕ, τ, nboot_train, δ, mean, λ, IS, old_ent, num_iters, rng)
    bound_fun(D, idxs, π, δ, tail) = nswildbst_CI(π, δ, tail, D, idxs, ϕ, τ, nboot_test, IS, rng)
    return opt_fun, bound_fun
end

function build_ns_testbs(D, π, oparams, ϕ, τ; nboot_train=200, nboot_test=500, δ=0.05, λ=0.01, IS=PerDecisionImportanceSampling(), old_ent=false, num_iters=100, rng=Base.GLOBAL_RNG)
    opt_fun(oparams, π, D, idxs) = maximize_ns_lower!(oparams, π, D, idxs, ϕ, τ, nboot_train, δ, mean, λ, IS, old_ent, num_iters, rng)
    bound_fun(D, idxs, π, δ, tail) = nswildbst_CI(π, δ, tail, D, idxs, ϕ, τ, nboot_test, IS, rng)
    return opt_fun, bound_fun
end

function off_policy_natgrad!(G, θ, Y, GY, ψ, F, π, D::BanditHistory{T,TA}, idxs, A, B, δ, num_boot, aggf, λ, IS::TI, rng) where {T,TA,TI<:UnweightedIS}
    set_params!(π, θ)
    compute_fisher!(F, ψ, π)
    estimate_entropyreturn!(Y, D, idxs, π, λ, IS)
    GY .= Zygote.gradient(y->mean(get_preds_and_residual(y, A, B)[1]), Y)[1]
    fill!(G, 0.0)

    for (i,idx) in enumerate(idxs)
        gradient_logp!(ψ, π, D.actions[idx])
        @. G += ψ * Y[i] * GY[i]
    end
    G ./= length(idxs)
    G .= inv(F) * G
end

function off_policy_natgrad!(G, θ, Y, GY, ψ, F, π::StatelessNormalPolicy, D::BanditHistory{T,TA}, idxs, A, B, δ, num_boot, aggf, λ, IS::TI, rng) where {T,TA,TI<:UnweightedIS}
    set_params!(π, θ)
    compute_fisher!(F, ψ, π)

    estimate_entropyreturn!(Y, D, idxs, π, λ, IS)
    GY .= Zygote.gradient(y->mean(get_preds_and_residual(y, A, B)[1]), Y)[1]

    fill!(G, 0.0)
    for (i,idx) in enumerate(idxs)
        logp = gradient_logp!(ψ, π, D.actions[idx])
        @. G += ψ * Y[i] * GY[i]
    end
    G ./= length(idxs)
    G .= inv(F) * G
end

function off_policy_natgrad_old!(G, θ, Y, GY, ψ, F, π::StatelessNormalPolicy, D::BanditHistory{T,TA}, idxs, A, B, δ, num_boot, aggf, λ, IS::TI, rng) where {T,TA,TI<:UnweightedIS}
    set_params!(π, θ)
    compute_fisher!(F, ψ, π)

    estimate_return!(Y, D, idxs, π, IS)
    GY .= Zygote.gradient(y->mean(get_preds_and_residual(y, A, B)[1]), Y)[1]

    fill!(G, 0.0)
    G .= λ .* gradient_entropy(π)
    for (i,idx) in enumerate(idxs)
        logp = gradient_logp!(ψ, π, D.actions[idx])
        @. G += ψ * Y[i] * GY[i]
    end
    G ./= length(idxs)
    G .= inv(F) * G
end

function ns_lower_grad(θ₀, D::BanditHistory{T,TA}, idxs, π, ϕ, τ, num_boot, δ, aggf, λ, IS::TI, old_ent, rng) where {T,TA,TI<:UnweightedIS}
    L = length(D)
    N = length(idxs)
    Y = zeros(T, N)
    GY = similar(Y)
    ψ = similar(θ₀)

    Φ, ϕτ = create_features(ϕ, idxs, collect(Float64, L+1:L+τ))
    A, B = get_coefs(Φ, ϕτ)

    F = zeros(T, (length(ψ), length(ψ)))

    if old_ent
        return (G, θ)->off_policy_natgrad_old!(G, θ, Y, GY, ψ, F, π, D, idxs, A, B, δ, num_boot, aggf, λ, IS, rng)
    else
        return (G, θ)->off_policy_natgrad!(G, θ, Y, GY, ψ, F, π, D, idxs, A, B, δ, num_boot, aggf, λ, IS, rng)
    end
end
