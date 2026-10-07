import pandas as pd
from scipy import stats
import math

df = pd.read_csv("../results/raw/results.csv")
print(df)

mean_time = df["wall_time"].mean()
std_time = df["wall_time"].std()
median_time = df["wall_time"].median()
min_time = df["wall_time"].min()
max_time = df["wall_time"].max()
num_runs = len(df)

print()
print("Mean wall-clock time:", mean_time, "seconds")
print("Standard deviation:", std_time, "seconds")
print("Median wall-clock time:", median_time, "seconds")
print("Minimum wall-clock time:", min_time, "seconds")
print("Maximum wall-clock time:", max_time, "seconds")
print("Number of runs:", num_runs)


standard_error = std_time / math.sqrt(num_runs)

t_critical = stats.t.ppf(0.975, df=num_runs - 1)

margin_of_error = t_critical * standard_error

ci_lower = mean_time - margin_of_error
ci_upper = mean_time + margin_of_error

print()
print("Standard error:", standard_error, "seconds")
print("95% confidence interval:", ci_lower, "to", ci_upper, "seconds")

summary = pd.DataFrame([{
    "num_runs": num_runs,
    "mean_wall_time": mean_time,
    "std_wall_time": std_time,
    "min_wall_time": min_time,
    "max_wall_time": max_time,
    "standard_error": standard_error,
    "ci_95_lower": ci_lower,
    "ci_95_upper": ci_upper
}])

summary.to_csv("../results/processed/summary.csv", index=False)
print()
print("Summary statistics saved to ../results/processed/summary.csv")