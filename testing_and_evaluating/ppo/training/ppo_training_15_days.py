# python -m testing_and_evaluating.ppo.training.ppo_training_15_days

import pandas as pd
import numpy as np
import random
import glob
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stock_exchange_env.trading_place_15_days import TradingEnvironment

files = glob.glob("synthetic_courses/training/*.csv")
choice = random.choice(files)
first_course = pd.read_csv(choice)["Price"]
env_start = Monitor(TradingEnvironment(df=first_course))

model = PPO(
    "MlpPolicy", env_start,
    verbose=1,
    tensorboard_log="./logs/15_days_models/ppo/activity_3",
)

gen_steps = 160_000
change_freq = 2_000
n = gen_steps // change_freq

for i in range(n):
    choice_random = random.choice(files)
    course = pd.read_csv(choice_random)["Price"]
    env_start = Monitor(TradingEnvironment(df=course))
    model.set_env(env_start)
    model.learn(total_timesteps=change_freq, reset_num_timesteps=False)

model.save("models/15_days_models/ppo/ppo_agent_v3")
print("finished training + saved model")