import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../results/raw/results.csv")

plt.plot(df["run"], df["wall_time"], marker='o')

plt.xlabel("Run")
plt.ylabel("Wall-clock time (seconds)")
plt.title("Wall-clock Time Across Repeated Runs")

plt.savefig("../figures/runtime_variability.png")

plt.show()