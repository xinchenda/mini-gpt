"""从实验 CSV 生成训练曲线和 Markdown 对照表。"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def choose_run(rows, requested):
    names = []
    for row in rows:
        if row["run_name"] not in names:
            names.append(row["run_name"])
    if requested is None:
        return names[-1]
    if requested not in names:
        choices = ", ".join(names)
        raise ValueError(f"找不到实验 {requested!r}，可选值为：{choices}")
    return requested


def write_summary_table(rows, path):
    columns = [
        ("run_name", "Run"),
        ("width", "Width"),
        ("layers", "Layers"),
        ("learning_rate", "LR"),
        ("warmup_steps", "Warmup"),
        ("run_steps", "Steps"),
        ("best_validation_loss", "Best val loss"),
        ("best_step", "Best step"),
        ("elapsed_seconds", "Seconds"),
        ("peak_memory_mb", "Peak MB"),
    ]
    lines = [
        "| " + " | ".join(title for _, title in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        values = [str(row.get(key, "")).replace("|", "\\|") for key, _ in columns]
        lines.append("| " + " | ".join(values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_run(rows, run_name, path):
    selected = [row for row in rows if row["run_name"] == run_name]
    if not selected:
        raise ValueError(f"实验 {run_name!r} 没有逐检查点数据")

    steps = [int(row["step"]) for row in selected]
    validation = [float(row["validation_loss"]) for row in selected]
    training_steps = []
    training_loss = []
    for row in selected:
        if row["training_batch_loss"]:
            training_steps.append(int(row["step"]))
            training_loss.append(float(row["training_batch_loss"]))

    learning_steps = []
    learning_rates = []
    for row in selected:
        if row["current_learning_rate"]:
            learning_steps.append(int(row["step"]))
            learning_rates.append(float(row["current_learning_rate"]))

    best_index = min(range(len(validation)), key=validation.__getitem__)
    best_step = steps[best_index]
    best_loss = validation[best_index]
    warmup_steps = int(selected[0]["warmup_steps"])

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(training_steps, training_loss, color="#9ca3af", linewidth=1.2,
                 label="Training batch loss")
    axes[0].plot(steps, validation, color="#2563eb", linewidth=2,
                 marker="o", markersize=3, label="Validation loss")
    axes[0].scatter([best_step], [best_loss], color="#dc2626", zorder=3,
                    label=f"Best validation loss: {best_loss:.3f} at step {best_step}")
    if warmup_steps:
        axes[0].axvline(warmup_steps, color="#f59e0b", linestyle="--",
                        label=f"Warmup ends: {warmup_steps}")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(learning_steps, learning_rates, color="#059669", linewidth=2)
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Learning rate")
    axes[1].grid(alpha=0.25)

    figure.suptitle(run_name)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("experiments/summary.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("experiments/metrics.csv"))
    parser.add_argument("--run-name", help="要绘图的实验名称，默认使用 metrics.csv 中最后一项")
    parser.add_argument("--plot", type=Path, default=Path("experiments/latest_training_curve.png"))
    parser.add_argument("--table", type=Path, default=Path("experiments/summary_table.md"))
    args = parser.parse_args()

    try:
        summary_rows = read_rows(args.summary)
        metric_rows = read_rows(args.metrics)
        run_name = choose_run(metric_rows, args.run_name)
        plot_run(metric_rows, run_name, args.plot)
        write_summary_table(summary_rows, args.table)
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))

    print(f"已生成曲线 {args.plot}")
    print(f"已生成对照表 {args.table}")
    print(f"绘图实验 {run_name}")


if __name__ == "__main__":
    main()
