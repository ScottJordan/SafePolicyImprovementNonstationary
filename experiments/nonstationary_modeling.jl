using Zygote: @adjoint
using Statistics
using LinearAlgebra

"""
    get_coefs(Φ, ϕτ)

returns least squares coefficients for making predictions
at observed points Φ and points of interest ϕτ.

This function is a helper function to be used for predicting
future points in a time series and for in wild bootstrap.
"""
function get_coefs(Φ, ϕτ)
    H = pinv(Φ' * Φ) * Φ'
    W = Φ * H
    ϕ = ϕτ * H
    return W, ϕ
end

"""
    get_preds_and_residual(Y, W, ϕ)

returns the baseline predictions using observed labels Y at points ϕ
and along with the vector of residuals Y - Ŷ.
W and ϕ should be the output of get_coefs.
"""
function get_preds_and_residual(Y, W, ϕ)
    Ŷ = W*Y
    y = ϕ * Ŷ
    ξ = Y .- Ŷ
    return y, ξ
end

"""
    wildbs_eval(y, ξ, ϕ, σ, f)

evaluates the prediction of a linear timeseries model using
noise generated for the wild bootstrap.

y is the prediction at points ϕ using original labels Y
ξ are the residual between the labels Y and prediction Ŷ
ϕ are the points (coefficients) at at which the predictions y are made
σ is the noise sample used to modify the residual, e.g., vector of {-1, 1}
f is the function to aggregate the result for all points in ϕ, e.g., sum, mean, maximum, minimum
"""
function wildbs_eval(y, ξ, ϕ, σ, f)
    return f(y .+ ϕ * (ξ .* σ))
end

"""
    wildbs_CI(y, ξ, ϕ, δ, num_boot, aggf)

computes the δ percentiles bootstrap using the wild bootstrap method with num_boot bootstrap samples
for original predictions y, with residuals ξ, at features ϕ, aggregated by aggf.
"""
function wildbs_CI(y, ξ, ϕ, δ, num_boot, aggf, rng)
    bsamples = [wildbs_eval(y, ξ, ϕ, sign.(randn(rng, length(ξ))), aggf) for i in 1:num_boot]
    return quantile(sort(bsamples), δ, sorted=true)
end

function get_coefst(Φ, ϕτ)
    H = pinv(Φ' * Φ) * Φ'
    W = Φ * H
    ϕ = ϕτ * H
    C = I - W
    return W, ϕ, C
end

function get_preds_and_residual_t(Y, W, ϕ, C)
    Ŷ = W*Y
    y = ϕ * Ŷ
    x = C * Ŷ
    ξ = Y .- Ŷ
    Σ = ϕ * Diagonal(ξ.^2) * ϕ'
    v = mean(Σ)
    
    return y, v, x, ξ
end

function wildbst_eval(x, C, ξ, ϕ, σ)
    ξ̂ = ξ .* σ
    ξ̂2 = (x .+ C * ξ̂).^2
    Δŷ = mean(ϕ * ξ̂)
    Σ = ϕ * Diagonal(ξ̂2) * ϕ'
    v̂ = mean(Σ)

    return Δŷ / √v̂
end


function bst_CIs(p, v, bsamples, δ)
    return p - quantile(sort(bsamples), 1.0-δ, sorted=true) * √v
end

function bst_CIs(p, v, bsamples, δ::Array{T,1}) where {T}
    return p .- quantile(sort(bsamples), 1.0 .- δ, sorted=true) .* √v
end

function wildbst_CI(y, v, x, ξ, ϕ, C, δ, num_boot,rng)
    bsamples = [wildbst_eval(x, C, ξ, ϕ, sign.(randn(rng, length(ξ)))) for i in 1:num_boot]
    p = mean(y)
    return p - quantile(sort(bsamples), 1.0-δ, sorted=true) * √v #bst_CIs(p, v, bsamples, δ)
end

# defines the adjoint function for sorting so Zygote can automatically differentiate the sort function.

@adjoint function sort(x)
     p = sortperm(x)
     x[p], x̄ -> (x̄[invperm(p)],)
end

"""
create features for time series using basis function ϕ,
for observed time points x, and future time points tau
"""
function create_features(ϕ, x, τ)
    Φ = hcat(ϕ.(x)...)'
    ϕτ = hcat(ϕ.(τ)...)'
    return Φ, ϕτ
end


function fourierseries(::Type{T}, order::Int) where {T}
    C = collect(T, 0:order) .* π
    return x -> @. cos(C * x)
end

function normalize_time(D, τ)
    return x -> x / (length(D) + τ)
end
