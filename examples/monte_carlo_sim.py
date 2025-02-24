import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.learning.quadrotor_environments import QuadrotorEnv
from rotorpy.learning.quadrotor_reward_functions import hover_reward
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.wind.dryden_winds import DrydenGust
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy

# Load models (with no wind and with wind)
def load_model(model_path, env, log_dir=None):
    return PPO.load(model_path, env=env, tensorboard_log=log_dir)

# Define function to calculate tracking error (Euclidean distance between desired and actual position)
def calculate_tracking_error(desired_positions, actual_positions):
    # Desired and actual positions should be 2D arrays with shape (num_timesteps, 3)
    errors = np.linalg.norm(desired_positions - actual_positions, axis=1)
    return errors

# Monte Carlo Simulation
def run_monte_carlo_simulation(models, num_trials=200, num_timesteps=200):
    tracking_errors_no_wind = []
    tracking_errors_with_wind = []
    
    for trial in range(num_trials):
        print(f"Running trial {trial+1} of {num_trials}...")

        # Initialize environments (with and without wind)
        env_no_wind = gym.make("Quadrotor-v0", control_mode='cmd_motor_speeds', reward_fn=hover_reward,
                               quad_params=quad_params, max_time=5, wind_profile=None, world=None, sim_rate=100,
                               render_mode=None, render_fps=60)
        env_with_wind = gym.make("Quadrotor-v0", control_mode='cmd_motor_speeds', reward_fn=hover_reward,
                                 quad_params=quad_params, max_time=5, wind_profile=DrydenGust(dt=1 / 100, sig_wind=np.array([75, 75, 30]), altitude=2.0),
                                 world=None, sim_rate=100, render_mode=None, render_fps=60)
        # Reset the environments
        obs_no_wind = env_no_wind.reset()[0]
        obs_with_wind = env_with_wind.reset()[0]

        # Initialize variables for tracking
        actual_positions_no_wind = []
        actual_positions_with_wind = []
        desired_positions = []  # Assuming you have some way of defining desired positions

        # Run the simulation for the given number of timesteps
        for t in range(num_timesteps):
            # Get actions from models (both no wind and with wind)
            action_no_wind, _ = models[0].predict(obs_no_wind, deterministic=True)
            action_with_wind, _ = models[1].predict(obs_with_wind, deterministic=True)

            # Step the environments
            obs_no_wind, _, done_no_wind, _, _ = env_no_wind.step(action_no_wind)
            obs_with_wind, _, done_with_wind, _, _ = env_with_wind.step(action_with_wind)

            # Append the actual positions to track
            actual_positions_no_wind.append(obs_no_wind[:3])  # Assuming positions are in first 3 elements
            actual_positions_with_wind.append(obs_with_wind[:3])

            # Append the desired positions (you need to define or generate these)
            desired_positions.append([0, 0, 1])  # Example desired position for hovering at (0, 0, 1)

            if done_no_wind and done_with_wind:
                break

        # Convert lists to numpy arrays
        actual_positions_no_wind = np.array(actual_positions_no_wind)
        actual_positions_with_wind = np.array(actual_positions_with_wind)
        desired_positions = np.array(desired_positions)

        # Calculate the tracking error for both models
        error_no_wind = calculate_tracking_error(desired_positions, actual_positions_no_wind)
        error_with_wind = calculate_tracking_error(desired_positions, actual_positions_with_wind)

        # Store the mean errors for each trial
        tracking_errors_no_wind.append(np.mean(error_no_wind))
        tracking_errors_with_wind.append(np.mean(error_with_wind))

    # Calculate the mean tracking error across all trials
    mean_error_no_wind = np.mean(tracking_errors_no_wind)
    mean_error_with_wind = np.mean(tracking_errors_with_wind)

    print(f"Mean Tracking Error (No Wind): {mean_error_no_wind}")
    print(f"Mean Tracking Error (With Wind): {mean_error_with_wind}")

    return mean_error_no_wind, mean_error_with_wind


# Define the path for loading trained models
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rotorpy", "learning", "policies", "PPO")
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rotorpy", "learning", "logs")

# hard coding the path for now
model_no_wind_path = os.path.join(models_dir, "19-27-36/hover_1000000.zip")  # path to no-wind model 
model_with_wind_path = os.path.join(models_dir, "18-57-51/hover_1000000.zip")  #s path to wind model

# Load the models
env_with_wind = gym.make("Quadrotor-v0", control_mode='cmd_motor_speeds', reward_fn=hover_reward,
                         quad_params=quad_params, max_time=5, wind_profile=DrydenGust(dt=1 / 100, sig_wind=np.array([75, 75, 30]), altitude=2.0),
                         world=None, sim_rate=100, render_mode=None, render_fps=60)

# Load the models
model_no_wind = load_model(model_no_wind_path, env_with_wind, log_dir)
model_with_wind = load_model(model_with_wind_path, env_with_wind, log_dir)

# Run the Monte Carlo simulation
run_monte_carlo_simulation([model_no_wind, model_with_wind], num_trials=1000)
