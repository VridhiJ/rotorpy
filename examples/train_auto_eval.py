import gymnasium as gym
import numpy as np
import os
from datetime import datetime
from eval import evaluate_model
from rotorpy.wind.dryden_winds import DrydenGust

from rotorpy.vehicles.crazyflie_params import quad_params  # Import quad params for the quadrotor environment.

# Import the QuadrotorEnv gymnasium environment using the following command.
from rotorpy.learning.quadrotor_environments import QuadrotorEnv

# Reward functions can be specified by the user, or we can import from existing reward functions.
from rotorpy.learning.quadrotor_reward_functions import hover_reward

"""
Here we check the progress of our training with periodic evaluations.

We will also track the agents under different wind conditions.

"""

# First we'll set up some directories for saving the policy and logs.
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rotorpy", "learning", "policies")
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rotorpy", "learning", "logs")
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Next import Stable Baselines.
try:
    import stable_baselines3
except:
    raise ImportError('To run this example you must have Stable Baselines installed via pip install stable_baselines3')

from stable_baselines3 import PPO                                   # We'll use PPO for training.
from stable_baselines3.ppo.policies import MlpPolicy                # The policy will be represented by an MLP

num_cpu = 4   # for parallelization

# Choose the weights for our reward function. Here we are creating a lambda function over hover_reward.
reward_function = lambda obs, act: hover_reward(obs, act, weights={'x': 1, 'v': 0.1, 'w': 0, 'u': 1e-5})



# Define training scenarios.
training_scenarios = {
    "No wind": {
        "wind_profile": None
    },
    "DrydenGust": {
        "wind_profile": DrydenGust(dt=1 / 100, sig_wind=np.array([75, 75, 30]), altitude=2.0)
    }
}

# Common environment parameters. For this demo we'll train a policy to command collective thrust and body rates.
# Turning render_mode="None" will make the training run much faster, as visualization is a current bottleneck.
env_config = {
    "id": "Quadrotor-v0",
    "control_mode": "cmd_motor_speeds",
    "reward_fn": lambda obs, act: hover_reward(obs, act, weights={'x': 1, 'v': 0.1, 'w': 0, 'u': 1e-5}),
    "quad_params": quad_params,
    "max_time": 5,
    "world": None,
    "sim_rate": 100,
    "render_mode": "None"
}


# Select a scenario
scenario_names = list(training_scenarios.keys())
for i, name in enumerate(scenario_names):
    print(f"{i}: {name}")
    
selected_id = int(input("Select scenario: "))
selected_scenario = training_scenarios[scenario_names[selected_id]]

# Merge selected scenario parameters into env_config
env_config.update(selected_scenario)

# Make the environment. 
env = gym.make(**env_config)

# Reset the environment
observation, info = env.reset(options={'initial_state': 'random','pos_bound': 2, 'vel_bound': 0})

# Create a new model
model = PPO(MlpPolicy, env, verbose=1, ent_coef=0.01, tensorboard_log=log_dir)

# Ask the user if they want to run evaluation periodically to see progress
auto_eval = input("Would you like to get training progress updates? (Y/N): ").strip().lower() == "y"

# If yes, ask user how frequently would they like to run the evaluation script
if auto_eval:
    eval_freq = int(input("How frequently would you like to run the evaluation? Enter a number: ").strip())
    

# Training...
num_timesteps = 20_000
num_epochs = 10

start_time = datetime.now()

epoch_count = 0
while True:  # Run indefinitely..

    # This line will run num_timesteps for training and log the results every so often.
    model.learn(total_timesteps=num_timesteps, reset_num_timesteps=False, tb_log_name="PPO-Quad_cmd-motor_"+start_time.strftime('%H-%M-%S'))

    # Save the model
    model_path = f"{models_dir}/PPO/{start_time.strftime('%H-%M-%S')}/hover_{num_timesteps*(epoch_count+1)}"
    model.save(model_path)

    # runs evaluation script periodically
    if auto_eval and epoch_count%(eval_freq) == 0: 
        evaluate_model(model_path, num_timesteps = num_timesteps*(epoch_count+1), selected_scenario = selected_scenario)

    epoch_count += 1
