"""绘制配对种子实验的逐种子结果、均值和标准差。"""

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("experiments/paired_seed_comparison.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/paired_seed_comparison.png"))
    parser.add_argument("--loss-label", default="Best Wind validation loss")
    args = parser.parse_args()

    with args.input.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = {}
    for row in rows:
        groups.setdefault(row["config"], []).append(row)

    figure, axis = plt.subplots(figsize=(8, 5.5))
    labels = []
    for index, (config, group) in enumerate(groups.items()):
        losses = [float(row["best_validation_loss"]) for row in group]
        offsets = [0.0] if len(group) == 1 else [
            -0.08 + 0.16 * i / (len(group) - 1) for i in range(len(group))
        ]
        axis.scatter([index + offset for offset in offsets], losses,
                     s=55, alpha=0.8, label=config if index == 0 else None)
        mean = statistics.mean(losses)
        std = statistics.stdev(losses) if len(losses) > 1 else 0
        axis.errorbar(index, mean, yerr=std, fmt="o", color="black",
                      capsize=8, markersize=7)
        labels.append(config)

    axis.set_xticks(range(len(labels)), labels)
    axis.set_ylabel(args.loss_label)
    axis.set_title("Paired seed comparison")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
