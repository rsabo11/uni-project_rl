# python -m synthetic_courses.generate.training_data

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

np.random.seed(10)

for i in range(8):
    steps = 500
    trend = np.random.normal(loc=0, scale=1, size=steps).cumsum()
    noise = np.random.normal(loc=0, scale=2, size=steps)
    price = 100 + trend + noise
    df = pd.DataFrame({"Price": price})
    df.to_csv(f"synthetic_courses/training/training_{i+1}.csv", index=False)

plt.figure(figsize=(10, 5))
for i in range(8):
    df = pd.read_csv(f"synthetic_courses/training/training_{i+1}.csv")
    plt.plot(df["Price"], label=f"Trend {i+1}")
plt.title("Training Stocks 1-8")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("synthetic_courses/training/plots/training_stocks_overview.png")
plt.close()