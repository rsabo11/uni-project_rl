import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnvironment(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, df):
        super(TradingEnvironment, self).__init__()

        self.df = df.reset_index(drop=True)
        self.actual_step = 0
        self.bank = 10000.0
        self.stock_owned = 0
        self.stock_price = 0.0
        self.prev_bank = self.bank
        self.action_space = spaces.Discrete(3)

        self.window_size = 5
        obs_dim = self.window_size + 4

        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    # later there could be more indicators for the agent to check regarding trends
    def _get_obs(self):
        start = max(0, self.actual_step - self.window_size + 1)
        prices = self.df.iloc[start : self.actual_step + 1].values.astype(np.float32)

        if len(prices) < self.window_size:
            pad = np.full(self.window_size - len(prices), prices[0], dtype=np.float32)
            prices = np.concatenate([pad, prices], axis=0)

        sma5  = float(self.df.iloc[max(0, self.actual_step - 4) : self.actual_step + 1].mean())
        sma20 = float(self.df.iloc[max(0, self.actual_step - 19) : self.actual_step + 1].mean())

        owned = float(self.stock_owned)
        bank  = float(self.bank)

        obs = np.concatenate([
            prices,
            [sma5, sma20],
            [owned, bank]
        ]).astype(np.float32)

        # double check
        assert obs.shape == (self.window_size + 4,), f"Wrong obs shape: {obs.shape}"

        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.actual_step   = 0
        self.bank        = 10000.0
        self.stock_owned    = 0
        self.prev_bank = self.bank

        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action):
        price_t = float(self.df.iloc[self.actual_step])

        if action == 1 and self.bank >= price_t:
            self.stock_owned += 1
            self.bank    -= price_t
        elif action == 2 and self.stock_owned > 0:
            self.stock_owned -= 1
            self.bank     += price_t

        net_before = self.bank + self.stock_owned * price_t
        self.actual_step += 1
        done = self.actual_step >= len(self.df) - 1
        price_tp1 = float(self.df.iloc[self.actual_step])
        self.stock_price = price_tp1
        
        net_after = self.bank + self.stock_owned * price_tp1
        reward = net_after - net_before
        self.prev_net_worth = net_after
        obs = self._get_obs()

        # additional reward
        move = (price_tp1 - price_t) / price_t
        if action == 1 and move > 0:
            reward += move
        elif action == 2 and move < 0:
            reward += -move



        return obs, reward, done, False, {}
    
    def render(self):
        print(f"Step: {self.actual_step}, Price: {self.stock_price:.2f}, Owned: {self.stock_owned}, Bank: {self.bank:.2f}")