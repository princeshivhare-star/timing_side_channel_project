import time
import os
import pandas as pd
import torch
from victim_model import create_model


def measure_inference_time(model, batch_size, repeats=100):
    x = torch.randn(batch_size, 784)

    # warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(repeats):
            _ = model(x)

    end = time.perf_counter()

    avg_time = (end - start) / repeats
    return avg_time


def experiment_batch_size():
    results = []
    model = create_model(num_layers=2)

    batch_sizes = [1, 4, 8, 16, 32, 64, 128]

    for batch in batch_sizes:
        avg_time = measure_inference_time(model, batch_size=batch)
        results.append({
            "experiment": "batch_size_leakage",
            "batch_size": batch,
            "num_layers": 2,
            "avg_time_sec": avg_time
        })

    return results


def experiment_model_depth():
    results = []
    batch_size = 32
    layer_options = [1, 2, 4, 8, 12]

    for layers in layer_options:
        model = create_model(num_layers=layers)
        avg_time = measure_inference_time(model, batch_size=batch_size)
        results.append({
            "experiment": "model_depth_leakage",
            "batch_size": batch_size,
            "num_layers": layers,
            "avg_time_sec": avg_time
        })

    return results


def save_results(results):
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("results/timing_results.csv", index=False)
    print(df)
    print("Results saved to results/timing_results.csv")
    return df


def main():
    print("Running timing side-channel experiments...")

    results = []
    results.extend(experiment_batch_size())
    results.extend(experiment_model_depth())

    save_results(results)


if __name__ == "__main__":
    main()