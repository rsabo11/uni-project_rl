# python -m tests.test_env

import pandas as pd
import numpy as np
import random

from stock_exchange_env.trading_place_15_days import TradingEnvironment

price_data = pd.Series(np.random.uniform(low=90, high=110, size=100))

env = TradingEnvironment(df=price_data)

obs, info = env.reset()
done = False
total_reward = 0

print("Start with random actions...\n")

while not done:
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward
    env.render()

print(f"\nTest completed. Full-Reward: {total_reward:.2f}")