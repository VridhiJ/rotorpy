import subprocess
import time
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from eval import evaluate_model

from rotorpy.vehicles.crazyflie_params import quad_params  # Import quad params for the quadrotor environment.

# Import the QuadrotorEnv gymnasium environment using the following command.
from rotorpy.learning.quadrotor_environments import QuadrotorEnv

# Reward functions can be specified by the user, or we can import from existing reward functions.
from rotorpy.learning.quadrotor_reward_functions import hover_reward

"""
In this script, we demonstrate how to train a hovering control policy in RotorPy using Proximal Policy Optimization. 
We use our custom quadrotor environment for Gymnasium along with stable baselines for the PPO implementation. 

The task is for the quadrotor to stabilize to hover at the origin when starting at a random position nearby. 

Training can be tracked using tensorboard, e.g. tensorboard --logdir=<log_dir>

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

# Make the environment. For this demo we'll train a policy to command collective thrust and body rates.
# Turning render_mode="None" will make the training run much faster, as visualization is a current bottleneck.
env = gym.make("Quadrotor-v0",
                control_mode ='cmd_motor_speeds',
                reward_fn = reward_function,
                quad_params = quad_params,
                max_time = 5,
                world = None,
                sim_rate = 100,
                render_mode='None')

# from stable_baselines3.common.env_checker import check_env
# check_env(env, warn=True)  # you can check the environment using built-in tools

# Reset the environment
observation, info = env.reset(options={'initial_state': 'random','pos_bound': 2, 'vel_bound': 0})

# Create a new model
model = PPO(MlpPolicy, env, verbose=1, ent_coef=0.01, tensorboard_log=log_dir)

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

    # runs evaluation script every 5 epochs
    if epoch_count%5 == 0: 
        evaluate_model(model_path)

    epoch_count += 1
    

    
    


'''
    if num_timesteps % 500000 == 0:
        # Run another file here
        script_path = "ppo_hover_eval.py"
        # Run the script using subprocess
        subprocess.run(["python", script_path])
'''

"""

training_model_num = "0"
flag = 0
subprocess_timer = time.time()
script_path = "ppo_hover_eval.py"

### within loop
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rotorpy", "learning", "policies",
                              "PPO")

    models_available = os.listdir(models_dir)
    if flag == 0:
        for i, name in enumerate(models_available):
            if name in "PPO-Quad_cmd-motor_" + start_time.strftime('%H-%M-%S'):
                training_model_num = str(i)
                flag = 1
# Check if 5 minutes have elapsed since the last subprocess run
    if time.time() - subprocess_timer >= 300:  # 300 seconds = 5 minutes

        # Open the subprocess and specify stdin=subprocess.PIPE to allow input
        process = subprocess.Popen(["python", script_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        # the inputs to be passed to the subprocess
        inputs = f"{training_model_num}\n{epoch_count}\n"

        # Pass the inputs to the subprocess using communicate()
        output, error = process.communicate(inputs)

        # Reset the subprocess timer
        subprocess_timer = time.time()
"""