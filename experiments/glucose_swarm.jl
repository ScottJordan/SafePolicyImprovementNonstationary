using Statistics
using EvaluationOfRLAlgs
using Random
using DataFrames
using CSV
using ArgParse
using Distributions


include("normalpolicy.jl")
include("softmaxpolicy.jl")
include("history.jl")
include("optimizers.jl")
include("offpolicy.jl")

include("environments.jl")
# include("glucose_env.jl")
include("nonstationary_modeling.jl")
include("nonstationary_pi.jl")

struct SafetyPerfRecord{T1, T2} <: Any where {T1, T2}
    t::Array{T1,1}
    Jpi::Array{T2,1}
    Jsafe::Array{T2,1}

    function SafetyPerfRecord(::Type{T1}, ::Type{T2}) where {T1,T2}
        new{T1,T2}(Array{T1,1}(), Array{T2,1}(), Array{T2,1}())
    end
end

function plot_results(rec::SafetyPerfRecord, rews, tidxs, piflag, title)
    p1 = plot(title=title)
    p2 = plot()
    #p3 = plot()
    issafety = zeros(length(rews))
    unsafe = zeros(length(rews))
    notnsf = zeros(length(rews))
    for (i, (ts, te)) in enumerate(tidxs)
        color = :crimson
        if piflag[i]
            color = :dodgerblue
            notnsf[ts:te] .= 1
            if mean(rec.Jsafe[ts:te]) > mean(rec.Jpi[ts:te])
                unsafe[ts:te] .= 1
            end
        end
        scatter!(p1, ts:te, rews[ts:te], markercolor=color, markersize=1, markerstrokewidth=0, label=nothing)
        # issafety[ts:te] .= piflag[i]
    end
    # rgrad = cgrad([:crimson, :dodgerblue])
    # plot!(p1, rec.t, rews, linestyle=:dot, lc=rgrad, line_z=issafety, label="observed rewards")
    plot!(p1, rec.t, rec.Jsafe, lc=:crimson, label="J(π_safe)")
    plot!(p1, rec.t, rec.Jpi, lc=:dodgerblue, label="J(π_c)", legend=:bottomleft)
    # plot!(p2, )
    xlabel!(p1, "Episode")
    ylabel!(p1, "Return")

    plot!(p2, rec.t, notnsf, label="Canditate Returned")
    plot!(p2, rec.t, unsafe, label="Unsafe Policy", legend=:bottomright)
    xlabel!(p2, "Episode")
    ylabel!(p2, "Probability")
    p = plot(p1, p2, layout=(2,1))
    savefig(p, "myplot.pdf")
    return p
end

function record_perf!(rec::SafetyPerfRecord, eval_fn, t, πc, πsafe)
    push!(rec.t, t)
    push!(rec.Jpi, eval_fn(πc))
    push!(rec.Jsafe, eval_fn(πsafe))
    # println(t, "\t", rec.Jpi[t], "\t", rec.Jsafe[t], "\t", get_params(πc))
end

function plot_results(rews, tidxs, piflag, title)
    p1 = plot(title=title)
    p2 = plot()
    #p3 = plot()
    issafety = zeros(length(rews))
    unsafe = zeros(length(rews))
    notnsf = zeros(length(rews))
    for (i, (ts, te)) in enumerate(tidxs)
        color = :crimson
        if piflag[i]
            color = :dodgerblue
            notnsf[ts:te] .= 1
        end
        scatter!(p1, ts:te, rews[ts:te], markercolor=color, markersize=1, markerstrokewidth=0, label=nothing)
        issafety[ts:te] .= piflag[i]
    end
    rgrad = cgrad([:crimson, :dodgerblue])
    plot!(p1, rec.t, rews, linestyle=:dot, lc=rgrad, line_z=issafety, label="observed rewards")
    # plot!(p1, rec.t, rec.Jsafe, lc=:crimson, label="J(π_safe)")
    # plot!(p1, rec.t, rec.Jpi, lc=:dodgerblue, label="J(π_c)", legend=:bottomleft)
    # plot!(p2, )
    xlabel!(p1, "Episode")
    ylabel!(p1, "Return")

    plot!(p2, rec.t, notnsf, label="Canditate Returned")
    # plot!(p2, rec.t, unsafe, label="Unsafe Policy", legend=:bottomright)
    xlabel!(p2, "Episode")
    ylabel!(p2, "Probability")
    p = plot(p1, p2, layout=(2,1))
    savefig(p, "myplot.pdf")
    return p
end

function load_eval(speed)
    mats = [CSV.File(joinpath(@__DIR__, "glucose_eval_data", "speed$(speed)_allact$(i).csv"), header=false) |> Tables.matrix for i in 1:5]
    means = [mean(mat, dims=1) for mat in mats]
    return mats, means
end

function optimize_nsglucose_safety(num_episodes, rng, speed, hyperparams, fpath, save_res)
    D = BanditHistory(Float64, Array{Float64,1})
    env = NSGlucoseSim(speed, abs(rand(rng, UInt32)))

    π = StatelessNormalPolicy(Float64, 2, 2.0, true)
    # θ = [20.0,30.0, 5.0, 5.0]
    θ = [20.0,30.0, 2.0, 2.0]
    set_params!(π, θ)
    πsafe = clone(π)


    env_fn(action, rng) = sample_reward!(env, action, rng)
    sample_fn(D, π, N) = collect_data!(D, π, env_fn, N, rng)

    oparams = AdamParams(get_params(π), 1e-2; β1=0.9, β2=0.999, ϵ=1e-5)

    τ, λ, opt_ratio, fborder, old_ent = hyperparams

    nboot_train = 200 # num bootstraps
    nboot_test = 500
    # τ = 8 # num_steps to optimize for future performance and to collect data for
    δ = 0.05 # percentile lower bound to maximize future for (use 1-ϵ for upper bound)
    # aggf = mean # function to aggregate future performance over (e.g., mean over τ steps,) maximium and minimum are also useful
    # λ = 0.000005
    IS = PerDecisionImportanceSampling()
    # num_opt_iters = 40

    num_opt_iters = round(Int, τ*opt_ratio)
    warmup_steps = 20
    sm = SplitLastKKeepTest(0.5)
    fb = fourierseries(Float64, fborder)
    nt = normalize_time(D, τ)
    ϕ(t) = fb(nt(t))

    num_iters = num_episodes / τ

    train_idxs = Array{Int, 1}()
    test_idxs = Array{Int, 1}()

    opt_fun, safety_fun = build_nsbst(D, π, oparams, ϕ, τ; nboot_train=nboot_train, nboot_test=nboot_test, δ=δ, λ=λ, IS=IS, old_ent=old_ent, num_iters=num_opt_iters, rng=rng)

    tidx, piflag = HICOPI!(oparams, π, D, train_idxs, test_idxs, sample_fn, opt_fun, safety_fun, πsafe, τ, δ, sm, num_iters, warmup_steps)
    # res = save_results(fpath, rec, D.rewards, tidx, piflag, save_res)
    res = save_results(fpath, D.rewards, tidx, piflag, save_res)
    # display(plot_results(rec, D.rewards, tidx, piflag, "NS Discrete Entropy"))
    return res
    # return (rec, D.rewards, tidx, piflag)
end

function nsglucose_safety(num_episodes, rng, speed, fpath, save_res)
    D = BanditHistory(Float64, Array{Float64,1})
    env = NSGlucoseSim(speed, abs(rand(rng, UInt32)))

    π = StatelessNormalPolicy(Float64, 2, 2.0, true)
    # θ = [20.0,30.0, 5.0, 5.0]
    θ = [20.0,30.0, 2.0, 2.0]
    set_params!(π, θ)
    πsafe = clone(π)


    env_fn(action, rng) = sample_reward!(env, action, rng)
    sample_fn(D, π, N) = collect_data!(D, π, env_fn, N, rng)

    warmup_steps = 20#0
    N = num_episodes + warmup_steps

    sample_fn(D, πsafe, N)
    tidx = [(1,N)]
    piflag = [false]

    res = save_results(fpath, D.rewards, tidx, piflag, save_res)
    # display(plot_results(rec, D.rewards, tidx, piflag, "NS Discrete Entropy"))
    return res
    # return (rec, D.rewards, tidx, piflag)
end

function optimize_disc_nsglucose_safety(num_episodes, rng, speed, hyperparams, fpath, save_res, seed)
    D = BanditHistory(Float64, Int)
    # env = NSDiscreteGlucoseSim(speed, abs(rand(rng, UInt32)))

    p = [0.5, 0.125, 0.125, 0.125, 0.125]
    θ = log.(p) .- mean(log.(p))
    θ .= exp.(θ) / sum(exp.(θ))

    π = StatelessSoftmaxPolicy(Float64, length(p))
    set_params!(π, θ)
    πsafe = clone(π)

    mats, eval_means = load_eval(speed)

    function eval_policy(t::Int, π) 
        J = 0.0
        p = π.probs
        for i in 1:5
            J += p[i] * eval_means[i][t]
        end
        return J
    end

    sample_counter = [0]
    # env_fn(action, rng) = sample_reward!(env, action, rng)
    function env_fn(action, rng)
        return mats[action][seed, sample_counter[1]]
    end
    
    
    rec = SafetyPerfRecord(Int, Float64)
    eval_fn(π) = eval_policy(sample_counter[1], π)
    function log_eval(action, rng, sample_counter, rec, eval_fn, π, πsafe)
        sample_counter[1] += 1
        record_perf!(rec, eval_fn, sample_counter[1], π, πsafe)
        return env_fn(action, rng)
    end
    log_fn = (action, rng) -> log_eval(action, rng, sample_counter, rec, eval_fn, π, πsafe)
    # sample_fn(D, π, N) = collect_data!(D, π, env_fn, N, rng)
    sample_fn(D, π, N) = collect_data!(D, π, log_fn, N, rng)
    # sample_fn(D, π, N) = collect_data!(D, π, env_fn, N, rng)

    oparams = AdamParams(get_params(π), 1e-2; β1=0.9, β2=0.999, ϵ=1e-5)

    τ, λ, opt_ratio, fborder, old_ent = hyperparams

    nboot_train = 200 # num bootstraps
    nboot_test = 500
    # τ = 8 # num_steps to optimize for future performance and to collect data for
    δ = 0.05 # percentile lower bound to maximize future for (use 1-ϵ for upper bound)
    # aggf = mean # function to aggregate future performance over (e.g., mean over τ steps,) maximium and minimum are also useful
    # λ = 0.000005
    IS = PerDecisionImportanceSampling()
    # num_opt_iters = 40

    num_opt_iters = round(Int, τ*opt_ratio)
    warmup_steps = 20
    sm = SplitLastKKeepTest(0.5)
    fb = fourierseries(Float64, fborder)
    nt = normalize_time(D, τ)
    ϕ(t) = fb(nt(t))

    num_iters = num_episodes / τ

    train_idxs = Array{Int, 1}()
    test_idxs = Array{Int, 1}()

    opt_fun, safety_fun = build_nsbst(D, π, oparams, ϕ, τ; nboot_train=nboot_train, nboot_test=nboot_test, δ=δ, λ=λ, IS=IS, old_ent=old_ent, num_iters=num_opt_iters, rng=rng)

    tidx, piflag = HICOPI!(oparams, π, D, train_idxs, test_idxs, sample_fn, opt_fun, safety_fun, πsafe, τ, δ, sm, num_iters, warmup_steps)
    res = save_results(fpath, rec, D.rewards, tidx, piflag, save_res)
    # res = save_results(fpath, D.rewards, tidx, piflag, save_res)
    # display(plot_results(rec, D.rewards, tidx, piflag, "NS Discrete Entropy"))
    return res
    # return (rec, D.rewards, tidx, piflag)
end

function disc_nsglucose_safety(num_episodes, rng, speed, fpath, save_res)
    D = BanditHistory(Float64, Int)
    env = NSDiscreteGlucoseSim(speed, abs(rand(rng, UInt32)))


    p = [0.5, 0.125, 0.125, 0.125, 0.125]
    θ = log.(p) .- mean(log.(p))
    θ .= exp.(θ) / sum(exp.(θ))

    π = StatelessSoftmaxPolicy(Float64, length(p))
    set_params!(π, θ)
    πsafe = clone(π)


    env_fn(action, rng) = sample_reward!(env, action, rng)
    sample_fn(D, π, N) = collect_data!(D, π, env_fn, N, rng)

    warmup_steps = 20#0
    N = num_episodes + warmup_steps

    sample_fn(D, πsafe, N)
    tidx = [(1,N)]
    piflag = [false]

    res = save_results(fpath, D.rewards, tidx, piflag, save_res)
    # display(plot_results(rec, D.rewards, tidx, piflag, "NS Discrete Entropy"))
    return res
    # return (rec, D.rewards, tidx, piflag)
end

function optimize_nscbandit_safety(num_episodes, rng, speed, hyperparams, fpath, save_res)
    D = BanditHistory(Float64, Array{Float64,1})
    # env = NSGlucoseSim(speed, rand(Int,rng))
    if speed == 0
        κ = 0.0
    elseif speed == 1
        κ = (2*pi)/2000.0
    elseif speed == 2
        κ = (2*pi)/1500.0
    elseif speed == 3
        κ = (2*pi)/1250.0
    else
        println("speed $speed not recognized")
        return nothing
    end
    env = NonstationaryQuadraticBanditParams(1.0, -10.0, 0.010, κ, [1.0])
    # params = [1.0, -10, 0.10] #-0.01(ax + b)^2 + c * randn()
    θ = zeros(2)
    θ[1] = 8
    θ[2] = 2
    π = StatelessNormalPolicy(Float64, 1, 1, true)
    set_params!(π, θ)
    πsafe = clone(π)

    # rng = MersenneTwister(0)


    env_fn(action, rng) = sample_reward!(env, action, rng)
    sample_counter = [0]
    rec = SafetyPerfRecord(Int, Float64)
    eval_fn(π) = eval_policy(env, π)
    function log_eval(action, rng, sample_counter, rec, eval_fn, π, πsafe)
        sample_counter[1] += 1
        record_perf!(rec, eval_fn, sample_counter[1], π, πsafe)
        println(env.t[1], " ", round(env.b*cos(env.t[1] * env.τ), digits=3), " ", round.(get_params(π), digits=3))
        return env_fn(action, rng)
    end
    log_fn = (action, rng) -> log_eval(action, rng, sample_counter, rec, eval_fn, π, πsafe)
    # sample_fn(D, π, N) = collect_data!(D, π, env_fn, N, rng)
    sample_fn(D, π, N) = collect_data!(D, π, log_fn, N, rng)

    iteration_counter = [0]
    # function sample_print(D, π, N)
        # sample_fn(D, π, N)
        # iteration_counter[1] += 1
        # println("iteration $(iteration_counter[1]), reward: $(mean(D.rewards[end-N+1:end])) policy: $(get_params(π))")
    # end

    # π = StatelessNormalPolicy(Float64, 2, 3.0, true)
    # θ = [20.0,30.0, 3.0, 3.0]
    # set_params!(π, θ)
    # πsafe = clone(π)


    # env_fn(action, rng) = sample_reward!(env, action, rng)
    # sample_fn(D, π, N) = collect_data!(D, π, env_fn, N, rng)

    oparams = AdamParams(get_params(π), 1e-2; β1=0.9, β2=0.999, ϵ=1e-5)

    τ, λ, opt_ratio, fborder, old_ent = hyperparams

    nboot_train = 200 # num bootstraps
    nboot_test = 500
    # τ = 8 # num_steps to optimize for future performance and to collect data for
    δ = 0.05 # percentile lower bound to maximize future for (use 1-ϵ for upper bound)
    # aggf = mean # function to aggregate future performance over (e.g., mean over τ steps,) maximium and minimum are also useful
    # λ = 0.000005
    IS = PerDecisionImportanceSampling()
    # num_opt_iters = 40

    num_opt_iters = round(Int, τ*opt_ratio)
    warmup_steps = 20
    sm = SplitLastKKeepTest(0.5)
    fb = fourierseries(Float64, fborder)
    nt = normalize_time(D, τ)
    ϕ(t) = fb(nt(t))

    num_iters = num_episodes / τ
    # ϕ(t) = fb(t/1520)
    # ϕ(t) = [1.0]
    train_idxs = Array{Int, 1}()
    test_idxs = Array{Int, 1}()

    # opt_fun, safety_fun = build_nsbs(D, π, oparams, ϕ, τ; nboot_train=nboot_train, nboot_test=nboot_test, δ=δ, aggf=aggf, λ=λ, IS=IS, num_iters=num_opt_iters, rng=rng)
    opt_fun, safety_fun = build_nsbst(D, π, oparams, ϕ, τ; nboot_train=nboot_train, nboot_test=nboot_test, δ=δ, λ=λ, IS=IS, old_ent=old_ent, num_iters=num_opt_iters, rng=rng)

    tidx, piflag = HICOPI!(oparams, π, D, train_idxs, test_idxs, sample_fn, opt_fun, safety_fun, πsafe, τ, δ, sm, num_iters, warmup_steps)
    # res = save_results(fpath, rec, D.rewards, tidx, piflag, save_res)
    # display(plot_results(rec, D.rewards, tidx, piflag, "NS Discrete Entropy"))
    # return res
    return (rec, D.rewards, tidx, piflag)
end

function combine_trials(results)
    T = length(results)
    N = length(results[1][2])
    t = results[1][1].t
    unsafe = zeros((N, T))
    notnsf = zeros((N, T))
    Jpi = zeros((N, T))
    Jalg = zeros((N, T))
    for (k, (rec, rews, tidxs, piflag)) in enumerate(results)
        unf = zeros(N)
        nnsf = zeros(N)
        for (i, (ts, te)) in enumerate(tidxs)
            if piflag[i]
                nnsf[ts:te] .= 1
                if mean(rec.Jsafe[ts:te]) > mean(rec.Jpi[ts:te])
                    unf[ts:te] .= 1
                end
            end
        end
        @. unsafe[:, k] = unf
        @. notnsf[:, k] = nnsf
        @. Jpi[:, k] += rec.Jpi
        @. Jalg[:, k] += nnsf * rec.Jpi + (1 - nnsf) * rec.Jsafe
    end

    mnunsafe = vec(mean(unsafe, dims=2))
    stdunsafe = vec(std(unsafe, dims=2)) ./ √T
    mnfound = vec(mean(notnsf, dims=2))
    stdfound = vec(std(notnsf, dims=2)) ./ √T
    mnJpi = vec(mean(Jpi, dims=2))
    stdJpi = vec(std(Jpi, dims=2)) ./ √T
    mnJalg = vec(mean(Jalg, dims=2))
    stdJalg = vec(mean(Jalg, dims=2)) ./ √T
    # println(size(mnunsafe), " ", size(stdunsafe), " ", size(unsafe))
    return t, (mnunsafe, stdunsafe), (mnfound, stdfound), (mnJpi, stdJpi), (mnJalg, stdJalg)
end

function learning_curves(res1, res2, baseline, labels, title)
    p1 = plot(title=title)
    p2 = plot()

    # rgrad = cgrad([:crimson, :dodgerblue])
    # plot!(p1, rec.t, rews, linestyle=:dot, lc=rgrad, line_z=issafety, label="observed rewards")
    t1, safe1, found1, Jpi1, Jalg1 = res1
    plot!(p1, t1, baseline, lc=:black, label="π_safe")
    plot!(p1, t1, Jalg1[1], ribbon=Jalg1[2], lc=:crimson, fillalpha=0.3, label=labels[1])
    plot!(p1, t1, Jpi1[1], lc=:crimson, linestyle=:dot, label=nothing)

    t2, safe2, found2, Jpi2, Jalg2 = res2
    plot!(p1, t2, Jalg2[1], ribbon=Jalg2[2], lc=:dodgerblue, fillalpha=0.3, label=labels[2])
    plot!(p1, t2, Jpi2[1], lc=:dodgerblue, linestyle=:dot, label=nothing, legend=:topleft)

    xlabel!(p1, "Episode")
    ylabel!(p1, "Performance")

    plot!(p2, t1, found1[1], ribbon=found1[2], linestyle=:dash, lc=:crimson, fillalpha=0.3, label="Canditate Returned-Baseline")
    plot!(p2, t1, safe1[1], ribbon=safe1[2], linestyle=:solid, lc=:crimson, fillalpha=0.3,  label="Unsafe Policy-Baseline")
    plot!(p2, t2, found2[1], ribbon=found2[2], linestyle=:dash, lc=:dodgerblue, fillalpha=0.3,  label="Canditate Returned-SPIN")
    plot!(p2, t2, safe2[1], ribbon=safe2[2], linestyle=:solid, lc=:dodgerblue, fillalpha=0.3,  label="Unsafe Policy-SPIN", legend=:topleft)

    xlabel!(p2, "Episode")
    ylabel!(p2, "Probability")
    p = plot(p1, p2, layout=(2,1))
    savefig(p, "learningcurve.pdf")
    return p
end

function save_results(fpath, rec::SafetyPerfRecord, rews, tidxs, piflag, save_res=true)
    unsafe = zeros(length(rews))
    notnsf = zeros(length(rews))
    for (i, (ts, te)) in enumerate(tidxs)
        if piflag[i]
            notnsf[ts:te] .= 1
            if mean(rec.Jsafe[ts:te]) > mean(rec.Jpi[ts:te])
                unsafe[ts:te] .= 1
            end
        end
    end
    df = DataFrame(t = rec.t, rews=rews, Jpi = rec.Jpi, found=notnsf, unsafe=unsafe)
    if save_res
        CSV.write(fpath, df)
    end
    obsperf = mean(rews)
    canperf = mean(rec.Jpi)
    algperf = mean(notnsf .* rec.Jpi .+ (1 .- notnsf) .* rec.Jsafe)
    foundpct = mean(notnsf)
    violation = mean(unsafe)
    regret = mean(notnsf .* rec.Jpi .+ (1 .- notnsf) .* rec.Jsafe .- rec.Jsafe)
    sumres = [obsperf, canperf, algperf, regret, foundpct, violation]
    return sumres
end

# function save_results(fpath, rews, tidxs, piflag, save_res=true)
#     notnsf = zeros(length(rews))
#     for (i, (ts, te)) in enumerate(tidxs)
#         if piflag[i]
#             notnsf[ts:te] .= 1
#         end
#     end
#     df = DataFrame(t = 1:length(rews), rews=rews, found=notnsf)
#     if save_res
#         CSV.write(fpath, df)
#     end
#     obsperf = mean(rews)
#     foundpct = mean(notnsf)

#     sumres = [obsperf, foundpct]
#     return sumres
# end

function logRand(low, high, rng)
    X = exp(rand(rng, Uniform(log(low), log(high))))
    logp = -log(log(high) - log(high) - log(X))
    return X, logp
end

function sample_ns_hyperparams(rng, old)
    τ = rand(rng, [2,4,6,8])
    λ = logRand(0.01, 1.0, rng)[1]
    opt_ratio = rand(rng)*3 + 2 #[2,5]
    fborder = rand(rng, 1:4)
    params = (round(Int, τ), λ, opt_ratio, round(Int, fborder), old)
    return params
end

function sample_stationary_hyperparams(rng, old)
    τ = rand(rng, [2,4,6,8])
    λ = logRand(0.01, 1.0, rng)[1]
    opt_ratio = rand(rng)*3 + 2 #[2,5]
    fborder = 0
    params = (round(Int, τ), λ, opt_ratio, round(Int, fborder), old)
    return params
end


function runsweep(id, seed, algname, save_dir, trials, speed, num_episodes)
    rng = Random.MersenneTwister(seed)

    save_name = "$(algname)_$(lpad(seed, 5, '0')).csv"


    save_path = joinpath(save_dir, save_name)

    open(save_path, "w") do f
        write(f, "tau,lambda,optratio,fborder,old_ent,obsperf,canperf,algperf,regret,foundpct,violation\n")
        # write(f, "tau,lambda,optratio,fborder,old_ent,obsperf,foundpct\n")
        flush(f)
        for trial in 1:trials
            rets_name = "returns_$(algname)_$(lpad(seed, 5, '0'))_$(lpad(trial, 5, '0')).csv"
            ret_path = joinpath(save_dir, rets_name)
            if algname == "stationary"
                hyps = sample_stationary_hyperparams(rng, false)
            elseif algname == "stationary-old"
                hyps = sample_stationary_hyperparams(rng, true)
            elseif algname == "nonstationary-old"
                hyps = sample_ns_hyperparams(rng, true)
            else
                hyps = sample_ns_hyperparams(rng, false)
            end
            # res = optimize_nsglucose_safety(num_episodes, rng, speed, hyps, ret_path, true)
            res = optimize_disc_nsglucose_safety(num_episodes, rng, speed, hyps, ret_path, true, seed)

            result = join([hyps..., res...], ',')
            write(f, "$(result)\n")
            flush(f)
            # println("$trial \t $(result)")
            # flush(stdout)
        end
    end

end

function runsafety(id, seed, save_dir, trials, speed, num_episodes)
    rng = Random.MersenneTwister(seed)

    save_name = "safety_$(lpad(seed, 5, '0')).csv"


    save_path = joinpath(save_dir, save_name)

    # open(save_path, "w") do f
        # write(f, "tau,lambda,optratio,fborder,obsperf,canperf,algperf,regret,foundpct,violation\n")
        # write(f, "tau,lambda,optratio,fborder,obsperf,foundpct\n")
        # flush(f)
    for trial in 1:trials
        rets_name = "returns_safety_$(lpad(seed, 5, '0'))_$(lpad(trial, 5, '0')).csv"
        ret_path = joinpath(save_dir, rets_name)

        res = nsglucose_safety(num_episodes, rng, speed, ret_path, true)
        res = disc_nsglucose_safety(num_episodes, rng, speed, ret_path, true)

        # result = join([hyps..., res...], ',')
        # write(f, "$(result)\n")
        # flush(f)
        # println("$trial \t $(result)")
        # flush(stdout)
    end

end

function tmp2(num_episodes, speed)
    # hyps = (Int(2), 0.06125, 3.0, Int(3))
    num_trials = 2
    hyps = (Int(4), 0.0001, 3.0, Int(3), false)
    rng = MersenneTwister(0)
    # optimize_nsglucose_safety(num_episodes, rng, speed, hyps, "nopath.csv", false)
    recs1 = [optimize_nscbandit_safety(num_episodes, rng, speed, hyps, "nopath.csv", false) for i in 1:num_trials]

    hyps = (Int(4), 0.0001, 3.0, Int(3), true)
    # rng = MersenneTwister(0)
    recs2 = [optimize_nscbandit_safety(num_episodes, rng, speed, hyps, "nopath.csv", false) for in in 1:num_trials]

    r1 = combine_trials(recs1)
    r2 = combine_trials(recs2)
    baseline = recs1[1][1].Jsafe
    # println(size.([r1[1],  r1[2][1], r1[2][2], r1[3][1], r1[3][2], r1[4][1], r1[4][2], r1[5][1], r1[5][2], baseline]))
    df1 = DataFrame(t = r1[1],  mnsafe = r1[2][1], stdsafe = r1[2][2], mnfound = r1[3][1], stdfound = r1[3][2], mnJpi = r1[4][1], stdJpi = r1[4][2], mnJalg = r1[5][1], stdJalg = r1[5][2], baseline=baseline)
    # CSV.write("nonstationary_learncurve.csv", df1)
    df2 = DataFrame(t = r2[1],  mnsafe = r2[2][1], stdsafe = r2[2][2], mnfound = r2[3][1], stdfound = r2[3][2], mnJpi = r2[4][1], stdJpi = r2[4][2], mnJalg = r2[5][1], stdJalg = r2[5][2], baseline=baseline)
    # CSV.write("stationary_learncurve.csv", df2)
    display(learning_curves(r2, r1, baseline, ["SPIN", "SPIN-OLD"], "Nonstationary Recommender System"))
    # display(learning_curves(r2, r1, baseline, ["Baseline", "SPIN"], "Nonstationary Recommender System"))
end

function main()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--alg-name", "-a"
            help = "name of the algorithm to run"
            arg_type = String
            required = true
        "--log-dir", "-l"
            help = "folder directory to store results"
            arg_type = String
            required = true
        "--id"
            help = "identifier for this experiment. Used in defining a random seed"
            arg_type = Int
            required = true
        "--seed", "-s"
            help = "random seed is seed + id"
            arg_type = Int
            required = true
        "--trials", "-t"
            help = "number of random trials to run"
            arg_type = Int
            default = 1
        "--speed"
            help = "speed of the environment [0,1,2,3]"
            arg_type = Int
            default = 1
        "--eps"
            help = "number of episodes to run"
            arg_type = Int
            default = 100

    end

    parsed_args = parse_args(ARGS, s)
    aname = parsed_args["alg-name"]
    println(aname)
    println(parsed_args["log-dir"])
    flush(stdout)
    save_dir = parsed_args["log-dir"]

    trials = parsed_args["trials"]
    id = parsed_args["id"]
    seed = parsed_args["seed"]
    speed = parsed_args["speed"]
    num_episodes = parsed_args["eps"]

    # save_dir = joinpath(save_dir, "glucose_$speed")
    save_dir = joinpath(save_dir, "dglucose_$speed")
    mkpath(save_dir)


    println(id, " ", seed, " ", id + seed)
    flush(stdout)
    seed = id + seed
    if aname == "safety"
        runsafety(id, seed, save_dir, trials, speed, num_episodes)
    else
        runsweep(id, seed, aname, save_dir, trials, speed, num_episodes)
    end

end

main()
