import time
import subprocess
import csv

results = []

for run in range(5):

    start_time = time.perf_counter()

    subprocess.run(["sleep", "2"]) # Replace this with ["python3", "some_ml_pipeline.py"]

    end_time = time.perf_counter()

    wall_time = end_time - start_time

    print("Run", run + 1, "- Wall-clock time:", wall_time, "seconds")

    results.append([run + 1, wall_time])

with open("../results/raw/results.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["run", "wall_time"])
    writer.writerows(results)