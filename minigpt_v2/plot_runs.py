"""Plot recorded validation curves without smoothing."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for run in args.runs:
        rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        ax.plot([row["step"] for row in rows],
                [row["validation_loss"] for row in rows], label=run.name, linewidth=1.2)
    ax.set(xlabel="Optimizer updates", ylabel="Fixed validation cross-entropy",
           title="MiniGPT v2 validation curves")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
