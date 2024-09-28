# GymnasiumWrapper.jl

This module is a Julia wrapper library for the popular [Farama Foundation's Gymnasium](https://gymnasium.farama.org) library. It provides a simple and intuitive interface for interacting with Gymnasium environments in Julia, allowing you to easily train and test reinforcement learning agents.

## Prerequisites

Before using GymnasiumWrapper.jl, make sure you have the following dependencies installed:

- Python: GymnasiumWrapper.jl requires Python to be installed on your system. You can download Python from the [official website](https://www.python.org).
- Gymnasium: Install the Gymnasium library in your Python environment using pip:
  ```bash
  pip install gymnasium
  ```
- NumPy: Install the NumPy library in your Python environment using pip:
  ```bash
  pip install numpy
  ```
- PyCall.jl: Install the PyCall.jl package in Julia to enable calling Python libraries. Refer to the [PyCall.jl documentation](https://github.com/JuliaPy/PyCall.jl) for detailed instructions on setting up PyCall with your Python environment.

## Installation

To use this module in your project, simply include the `GymnasiumWrapper.jl` file in your project directory and import it using:

```julia
include("Gymnasium.jl")
using .GymnasiumWrapper
```

or

Install with Pkg:

```julia
using Pkg
Pkg.add(url="https://github.com/p-w-rs/GymnasiumWrapper.jl")

using GymnasiumWrapper
```

## Usage

### Creating an Environment

You can create a Gymnasium environment using the `Env` function and specifying the environment name:

```julia
env = Env("CartPole-v1")
```

If you want to create an environment with a flattened observation space, you can use the `FlatEnv` function:

```julia
env = FlatEnv("CarRacing-v2")"
```

It also supports creating and AtariEnv environment with a resized observation space:

```julia
env = AtariEnv("ALE/Breakout-v5")
```

### Interacting with the Environment

- `reset!(env)`: Reset the environment and return the initial state.
- `step!(env, action)`: Take a step in the environment with the given action and return the next state, reward, and done flag.
- `render!(env)`: Render the environment.
- `close!(env)`: Close the environment.

Here's an example of how to interact with the environment:

```julia
state = reset!(env)
done = false
while !done
    action = sample_action(env)
    next_state, reward, done = step!(env, action)
    render!(env)
    state = next_state
end
close!(env)
```

### Accessing Environment Properties

- `env.s_dim`: The dimension of the state space.
- `env.s_space`: The state space object.
- `env.a_dim`: The dimension of the action space.
- `env.a_space`: The action space object.

You can use these properties to obtain information about the environment and design your reinforcement learning algorithms accordingly.

## Credits

Gym.jl is a wrapper library for the [Farama Foundation's Gymnasium](https://gymnasium.farama.org) library. We would like to acknowledge and express our gratitude to the developers and contributors of Gymnasium for their excellent work in creating and maintaining the library.
