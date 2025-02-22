import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.learning.quadrotor_environments import QuadrotorEnv
from rotorpy.learning.quadrotor_reward_functions import hover_reward
from rotorpy.controllers.quadrotor_control import SE3Control
from stable_baselines3 import PPO

"""
Tracking the 10 agents for 2 seconds.
"""

def evaluate_model(model_path, num_quads=10):
    """
    Evaluates the trained policy for hovering using PPO.
    :param model_path: Path to the trained model files
    :param num_quads: Number of quadrotors to evaluate
    """

    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)

    # Define reward function
    reward_function = lambda obs, act: hover_reward(obs, act, weights={'x': 1, 'v': 0.1, 'w': 0, 'u': 1e-5})

    # Set up figure for visualization
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    def make_env():
        return gym.make("Quadrotor-v0", 
                        control_mode='cmd_motor_speeds', 
                        reward_fn=reward_function,
                        quad_params=quad_params,
                        max_time=2,
                        world=None,
                        sim_rate=100,
                        render_mode='3D',
                        render_fps=60,
                        fig=fig,
                        ax=ax,
                        color='b')

    envs = [make_env() for _ in range(num_quads)]
    envs.append(gym.make("Quadrotor-v0", 
                          control_mode='cmd_motor_speeds', 
                          reward_fn=reward_function,
                          quad_params=quad_params,
                          max_time=2,
                          world=None,
                          sim_rate=100,
                          render_mode='3D',
                          render_fps=60,
                          fig=fig,
                          ax=ax,
                          color='k'))

    observations = [env.reset()[0] for env in envs]
    terminated = [False] * len(observations)

    while not all(terminated):
        for i, env in enumerate(envs):
            env.render()

            if i == len(envs) - 1:
                # Use baseline SE3 controller
                baseline_controller = SE3Control(quad_params)
                state = {'x': observations[i][0:3], 'v': observations[i][3:6], 'q': observations[i][6:10], 'w': observations[i][10:13]}
                flat = {'x': [0, 0, 0], 'x_dot': [0, 0, 0], 'x_ddot': [0, 0, 0], 'x_dddot': [0, 0, 0], 'yaw': 0, 'yaw_dot': 0, 'yaw_ddot': 0}
                control_dict = baseline_controller.update(0, state, flat)
                action = np.interp(control_dict['cmd_motor_speeds'], [env.unwrapped.rotor_speed_min, env.unwrapped.rotor_speed_max], [-1, 1])
            else:
                action, _ = model.predict(observations[i], deterministic=True)

            observations[i], _, terminated[i], _, _ = env.step(action)
    
    plt.show()
