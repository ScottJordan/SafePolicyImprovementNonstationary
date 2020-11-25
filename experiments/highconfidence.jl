using EvaluationOfRLAlgs

"""
    safety_lastk_split(D, K)

Split the data D up into train and test sets by
randomly choosing half of the last K samples to be
in the test set and the rest in the train. This is
useful when data in D has been used to make policy
improvements in the past, but the K samples are new.

# Examples
```julia-repl
#D is a list of N trajectories
#π is a policy to collect data
K = 10# is the number of episodes to collect
collect_data!(D, π, K)
train_idxs, test_idxs = safety_lastk_split(D, K)
D[train_idxs]; returns the list of 1:N plus half of the last K samples
D[test_idxs]; returns half the last K samples
```
"""
function safety_lastk_split(D, K)
    N = length(D)
    idxs = randperm(K)
    k = floor(Int, K/2)
    train = vcat(1:N, N .+ idxs[1:k])
    test = N .+ idxs[k+1:end]
    return train, test
end


"""
    HICOPI_safety_test

This function performs a high confidence safety test of policy π
in reference to a policy πsafe. The evaluation function f computes
the confidence interval for a policy and level δ. The either
returns a symbol indicating if π :safe or :uncertain. :uncertain
means that it cannot be gauranteed that π is better the πsafe.
"""
function HICOPI_safety_test(f, π, πsafe, δ)
    pilow = f(π, δ/2.0, :left)
    safehigh = f(πsafe, δ/2.0, :right)
    # println(round(pilow-safehigh, digits=3), "\t", round.(π.probs, digits=2), "\t", round.(get_params(π), digits=3))#, " ", get_params(πsafe))
    # println(round(pilow-safehigh, digits=3), "\t", round.(get_params(π), digits=3))#, " ", get_params(πsafe))
    # println("π low, πsafe high: $pilow, $safehigh")
    if pilow > safehigh
        return :safe
    else
        return :uncertain
    end
end

"""
    HICOPI(D, sample_fn!, optimize_fn, confidence_test, π, πsafe, τ, δ)

TODO update this description
This is a high confidence off-policy policy improvement function that
performs a single iteration of collecting τ data samples with policy π,
finding a canditate policy, πc, and then checking to see if πc is
better than the safe policy, πsafe, with confidence δ. If πc is better
return πc otherwise return :NSF no solution found.

This funciton is generic and takes as input:
D, previous data, which is possibly empty
sample_fn!, a function to sample new data points using policy π
optimize_fn, a funciton to find a new policy on data D
confidence_test, a function that computes a high confidence upper or lower bound on a policies performance
π, a policy to collect data with and initialize the optimization search with
πsafe, a policy that is consider a safe baseline that can always be trusted
τ, number of samples to collect data for. τ could represent the number of episodes or draws from a bandit.
δ, confidence level to use to ensure that safe policies or :NSF is return with probability at least 1-δ.
"""
function HICOPI_step!(oparams, π, D, train_idxs, test_idxs, optimize_fn!, confidence_bound, πsafe, δ)
    optimize_fn!(oparams, π, D, train_idxs)

    # If there is no test-data, then ignore the safety test_idxs
    # This can only happen when split ratio of train:test = 100:0
    if length(test_idxs) < 1
        result = :safe
    else
        result = HICOPI_safety_test((π, δ, tail) -> confidence_bound(D, test_idxs, π, δ, tail), π, πsafe, δ)
    end

    if result == :uncertain
        return :NSF
    else
        return π
    end
end

function HICOPI!(oparams, π, D, train_idxs, test_idxs, sample_fn!, optimize_fn!, confidence_bound, πsafe, τ, δ, split_method, num_iterations, warmup_steps)
    πbehavior = πsafe

    timeidxs = Array{Tuple{Int,Int},1}()
    picflag = Array{Bool, 1}()
    using_pic = false
    push!(timeidxs, (1,warmup_steps))
    push!(picflag, false)
    collect_and_split!(D, train_idxs, test_idxs, πbehavior, warmup_steps, sample_fn!, split_method)

    for i in 1:num_iterations
        n = length(D)
        push!(timeidxs, (n+1, n+τ))
        push!(picflag, using_pic)
        collect_and_split!(D, train_idxs, test_idxs, πbehavior, τ, sample_fn!, split_method)
        result = HICOPI_step!(oparams, π, D, train_idxs, test_idxs, optimize_fn!, confidence_bound, πsafe, δ)
        if result == :NSF
            πbehavior = πsafe
            # println("iteration $i π is not safe")
            using_pic = false
        else
            πbehavior = π
            # println("iteration $i π is safe")
            using_pic = true
        end
    end
    return timeidxs, picflag
end

abstract type AbstractSplitMethod end

#unbaised method for splitting
struct SplitLastK{T} <: AbstractSplitMethod where {T}
    p::T
end

# biased method for splitting
struct SplitLastKKeepTest{T} <: AbstractSplitMethod where {T}
    p::T
end


function collect_and_split!(D, train_idxs, test_idxs, π, N, sample_fn!, split_method::SplitLastK)
    L = length(D)
    sample_fn!(D, π, N)
    idxs = randperm(N)
    k = floor(Int, split_method.p*N)
    # append!(train_idxs, N .+ idxs[1:k])
    empty!(train_idxs)
    empty!(test_idxs)
    append!(train_idxs, 1:L)
    append!(train_idxs, L .+ idxs[1:k])
    append!(test_idxs, L .+ idxs[k+1:end])
end

function collect_and_split!(D, train_idxs, test_idxs, π, N, sample_fn!, split_method::SplitLastKKeepTest)
    L = length(D)
    sample_fn!(D, π, N)
    idxs = randperm(N)
    k = floor(Int, split_method.p*N)
    append!(train_idxs, L .+ idxs[1:k])
    append!(test_idxs, L .+ idxs[k+1:end])

end
