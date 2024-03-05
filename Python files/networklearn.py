import os
import sys
import time
import numpy as np
from stable_baselines3 import PPO, A2C, SAC
from sb3_contrib import ARS, TRPO
from stable_baselines3.common.logger import configure
from network_management import NetInvMgmtMasterEnv

models_dir = f"RL_models/A2C_3/models/" # Change algorithm path here for models directory.
logdir = f"RL_models/A2C_3/logs/"		# Change algorithm path here for logs directory.

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = NetInvMgmtMasterEnv()
env.reset()

tmp_path = r"D:\Files\Data science\Learning\Reinforcement Learning\NetworkRL\RL_models\A2C_3\CSV_logs" # Change algorithm path here for csv logger directory.

new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

model = A2C('MlpPolicy', env, verbose=1) # Change RL-algorithm here which will be used in the training phase.


model.set_logger(new_logger)
model.learn(total_timesteps = 2000000)

# Code to run in the Command prompt:
# cd D:\Files\Data science\Learning\Reinforcement Learning\NetworkRL
# python3 networklearn.py

# If needed: Open Jupyter Notebook from a Drive D
# jupyter notebook --notebook-dir=D:/