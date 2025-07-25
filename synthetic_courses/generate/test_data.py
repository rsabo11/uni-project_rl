# python -m synthetic_courses.generate.test_data

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

np.random.seed(123)

for i in range(5):
    steps = 500
    trend = np.random.normal(loc=0, scale=1, size=steps).cumsum()
    noise = np.random.normal(loc=0, scale=2, size=steps)
    price = 100 + trend + noise
    df = pd.DataFrame({"Price": price})
    df.to_csv(f"synthetic_courses/tests/test_{i+1}.csv", index=False)

plt.figure(figsize=(10, 5))
for i in range(5):
    df = pd.read_csv(f"synthetic_courses/tests/test_{i+1}.csv")
    plt.plot(df["Price"], label=f"Test {i+1}")
plt.title("Test Stocks 1-5")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("synthetic_courses/tests/plots/test_stocks_overview.png")
plt.close()