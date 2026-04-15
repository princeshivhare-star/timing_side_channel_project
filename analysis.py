import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_batch_size_leakage(df):
    subset = df[df["experiment"] == "batch_size_leakage"]

    plt.figure(figsize=(8, 5))
    plt.plot(subset["batch_size"], subset["avg_time_sec"], marker="o")
    plt.xlabel("Batch Size")
    plt.ylabel("Average Inference Time (sec)")
    plt.title("Timing Leakage vs Batch Size")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/batch_size_leakage.png")
    plt.show()


def plot_model_depth_leakage(df):
    subset = df[df["experiment"] == "model_depth_leakage"]

    plt.figure(figsize=(8, 5))
    plt.plot(subset["num_layers"], subset["avg_time_sec"], marker="o")
    plt.xlabel("Number of Layers")
    plt.ylabel("Average Inference Time (sec)")
    plt.title("Timing Leakage vs Model Depth")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/model_depth_leakage.png")
    plt.show()


def main():
    df = pd.read_csv("results/timing_results.csv")
    plot_batch_size_leakage(df)
    plot_model_depth_leakage(df)
    print("Plots saved in results folder")


if __name__ == "__main__":
    main()