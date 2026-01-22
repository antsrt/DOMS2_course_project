#!/usr/bin/env python3
import argparse
import csv
import os

import matplotlib.pyplot as plt


def load_csv(path):
    speeds = []
    cots = []
    counts = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            speeds.append(float(row["speed"]))
            cots.append(float(row["cot_mean"]))
            count_val = row.get("count")
            if count_val is None or count_val == "":
                counts.append(None)
            else:
                counts.append(int(float(count_val)))
    return speeds, cots, counts


def compute_mean(cots, counts, weighted):
    if not cots:
        return 0.0
    if not weighted:
        return sum(cots) / len(cots)
    if any(c is None for c in counts):
        return sum(cots) / len(cots)
    total = sum(counts)
    if total <= 0:
        return sum(cots) / len(cots)
    return sum(c * w for c, w in zip(cots, counts)) / total


def plot_cot_csv(path, weighted, save_path=None):
    speeds, cots, counts = load_csv(path)
    pairs = sorted(zip(speeds, cots, counts), key=lambda x: x[0])
    speeds = [p[0] for p in pairs]
    cots = [p[1] for p in pairs]
    counts = [p[2] for p in pairs]

    mean_cot = compute_mean(cots, counts, weighted)

    fig, ax = plt.subplots()
    ax.plot(speeds, cots, marker="o", label="cot_mean per bin")
    ax.axhline(mean_cot, color="r", linestyle="--", label=f"mean {mean_cot:.3f}")
    ax.set(xlabel="speed [m/s]", ylabel="CoT", title="Cost of Transport vs Speed")
    ax.legend()
    fig.tight_layout()

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot CoT vs speed from CSV.")
    parser.add_argument("csv_path", help="Path to CSV file (speed,cot_mean[,count]).")
    parser.add_argument("--weighted", action="store_true", help="Use weighted mean when count is present.")
    parser.add_argument("--save", dest="save_path", help="Save figure to file instead of showing.")
    args = parser.parse_args()

    plot_cot_csv(args.csv_path, args.weighted, args.save_path)


if __name__ == "__main__":
    main()
