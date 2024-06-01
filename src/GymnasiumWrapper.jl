module GymnasiumWrapper

export Env, FlatEnv, AtariEnv, sample_action, reset!, step!, render!, close!

using PyCall

np = pyimport("numpy")
gym = pyimport("gymnasium")
gym_wrappers = pyimport("gymnasium.wrappers")

abstract type AbstractEnv end

struct Env <: AbstractEnv
    env
    s_dim
    s_space
    a_dim
    a_space
    is_discrete
    env_name
    kwargs
end

function Env(env_name::String; kwargs...)::Env
    env = gym.make(env_name; kwargs...)
    s_dim, s_space = get_state_dim_and_space(env)
    a_dim, a_space, is_discrete = get_action_dim_and_space(env)
    return Env(env, s_dim, s_space, a_dim, a_space, is_discrete, env_name, kwargs)
end

function FlatEnv(env_name::String; kwargs...)::Env
    env = gym_wrappers.FlattenObservation(gym.make(env_name; kwargs...))
    s_dim, s_space = get_state_dim_and_space(env)
    a_dim, a_space, is_discrete = get_action_dim_and_space(env)
    return Env(env, s_dim, s_space, a_dim, a_space, is_discrete, env_name, kwargs)
end

function AtariEnv(env_name::String; kwargs...)::AtariEnv
    env = gym.make(env_name; frameskip=1, kwargs...)
    env = gym_wrappers.AtariPreprocessing(env, frame_skip=4, grayscale_obs=true, screen_size=84, terminal_on_life_loss=false)
    s_dim, s_space = get_state_dim_and_space(env)
    a_dim, a_space, is_discrete = get_action_dim_and_space(env)
    return Env(env, s_dim, s_space, a_dim, a_space, is_discrete, env_name, kwargs)
end

function get_state_dim_and_space(env)
    if pyisinstance(env.observation_space, gym.spaces.Box)
        low = np.asarray(env.observation_space.low)
        high = np.asarray(env.observation_space.high)
        s_dim = size(low)
        s_space = (low, high)
    elseif pyisinstance(env.observation_space, gym.spaces.Discrete)
        s_dim = (env.observation_space.n,)
        s_space = env.observation_space.n
    else
        error("Unsupported observation space type: $(typeof(env.observation_space))")
    end
    return s_dim, s_space
end

function get_action_dim_and_space(env)
    if pyisinstance(env.action_space, gym.spaces.Box)
        low = np.asarray(env.action_space.low)
        high = np.asarray(env.action_space.high)
        a_dim = size(low)
        a_space = (low, high)
        is_discrete = false
    elseif pyisinstance(env.action_space, gym.spaces.Discrete)
        a_dim = (env.action_space.n,)
        a_space = env.action_space.n
        is_discrete = true
    else
        error("Unsupported action space type: $(typeof(env.action_space))")
    end
    return a_dim, a_space, is_discrete
end

function sample_action(env::AbstractEnv)
    env.env.action_space.sample()
end

function reset!(env::AbstractEnv)
    state, info = env.env.reset()
    done = false
    return state, done
end

function step!(env::AbstractEnv, action)
    state, reward, terminal, trunc, info = env.env.step(action)
    done = terminal || trunc
    return state, reward, done
end

function render!(env::AbstractEnv)
    env.env.render()
end

function close!(env::AbstractEnv)
    env.env.close()
end

end # module GymnasiumWrapper
