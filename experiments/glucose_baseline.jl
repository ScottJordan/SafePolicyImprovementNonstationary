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
include("glucose_env.jl")
include("nonstationary_modeling.jl")
include("nonstationary_pi.jl")

function collect_data!(H::BanditHistory, action::Int, sample_fn!, N, rng)
    for i in 1:N
        logp = 0.0
        r = sample_fn!(action, rng)
        push!(H, deepcopy(action), logp, r)
    end
end


function disc_nsglucose_eval(action::Int, num_episodes, rng, speed)
    D = BanditHistory(Float64, Int)
    env = NSDiscreteGlucoseSim(speed, abs(rand(rng, UInt32)))

    env_fn(action, rng) = sample_reward!(env, action, rng)
    

    warmup_steps = 20#0
    N = num_episodes + warmup_steps

    collect_data!(D, action, env_fn, N, rng)

    return D.rewards
end

function runsweep(id, seed, algname, save_dir, trials, speed, num_episodes)
    rng = Random.MersenneTwister(seed)

    save_name = "$(algname)_$(lpad(seed, 5, '0')).csv"


    save_path = joinpath(save_dir, save_name)

    open(save_path, "w") do f
        
        for trial in 1:trials
            res = disc_nsglucose_eval(algname, num_episodes, rng, speed)

            result = join(res, ',')
            write(f, "$(result)\n")
            flush(f)
        end
    end

end


function main()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--action", "-a"
            help = "action to evaluate"
            arg_type = Int
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
    aname = parsed_args["action"]
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
    save_dir = joinpath(save_dir, "dglucose_eval_$speed")
    println("path $save_dir")
    mkpath(save_dir)


    println(id, " ", seed, " ", id + seed)
    flush(stdout)
    seed = id + seed
    runsweep(id, seed, aname, save_dir, trials, speed, num_episodes)
end

main()
