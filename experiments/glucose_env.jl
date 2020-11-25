using PyCall
include("environments.jl")
gsim = pyimport("simglucose")

struct NSGlucoseSim <: AbstractContinuousBandit
    env::PyObject

    function NSGlucoseSim(speed::Int, seed)
        env = gsim.envs.NS_T1DSimEnv(speed=speed, oracle=-1, seed=seed)
        new(env)
    end
end

struct NSDiscreteGlucoseSim <: AbstractDiscreteBandit
    env::PyObject
    actions::Array{Tuple{Float64,Float64},1}

    function NSDiscreteGlucoseSim(speed::Int, seed)
        env = gsim.envs.NS_T1DSimEnv(speed=speed, oracle=-1, seed=seed)
        actions = Array{Tuple{Float64,Float64},1}([(23.0,33.0,), (19.0,26.0), (14.0,21.0), (9.0,16.0), (4.0,11.0)])
        new(env, actions)
    end
end

function sample_reward!(b::NSGlucoseSim, action::Array{Float64,1}, rng=nothing) where {T}
    b.env.reset()
    _, r, _, _ = b.env.step(action)
    return convert(Float64,r)
end

function sample_reward!(b::NSDiscreteGlucoseSim, action::Int, rng=nothing) where {T}
    b.env.reset()
    _, r, _, _ = b.env.step(b.actions[action])
    return convert(Float64,r)
end
