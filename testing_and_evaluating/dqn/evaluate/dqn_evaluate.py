# python -m testing_and_evaluating.dqn.evaluate.dqn_evaluate

import pandas as pd
from stable_baselines3 import DQN
from stock_exchange_env.trading_place_5_days import TradingEnvironment
from stable_baselines3.common.monitor import Monitor

df = pd.read_csv("synthetic_courses/tests/test_5.csv")["Price"]

env = Monitor(TradingEnvironment(df=df))

model = DQN.load("models/dqn/dqn_agent_v1", env=env)

obs, _ = env.reset()
done = False

action_counts = {0: 0, 1: 0, 2: 0}

total_reward = 0.0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    action_counts[int(action)] += 1
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward

total_value = env.env.bank + env.env.stock_owned * env.env.stock_price

# Results
print(f"▶ Bank: {env.env.bank:.2f} €")
print(f"▶ #Stocks: {env.env.stock_owned}")
print(f"▶ Stock Value: {env.env.stock_price} €")
print(f"▶ Value of Stocks: {env.env.stock_owned * env.env.stock_price:.2f} €")
print(f"▶ Actions: Hold={action_counts[0]}, Buy={action_counts[1]}, Sell={action_counts[2]}")
print(f"▶ Total Value: {total_value:.2f} €")
print(f"▶ Total Reward: {total_reward:.2f}")