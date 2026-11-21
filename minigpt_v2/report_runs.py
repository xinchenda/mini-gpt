"""Summarize completed runs and select using validation only."""
import argparse
import json
from pathlib import Path


def summarize(path):
    rows = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
    provenance = json.loads((path / "provenance.json").read_text())
    protocol = provenance["protocol"]
    if rows[-1]["step"] != protocol["steps"]:
        raise ValueError(f"Run incomplete: {path}")
    best = min(rows, key=lambda row: row["validation_loss"])
    return dict(run=str(path), config=protocol["config"], seed=protocol["seed"],
                trained_steps=rows[-1]["step"], best_step=best["step"],
                best_validation_loss=best["validation_loss"],
                tokens_per_second=rows[-1]["tokens_per_second"],
                training_target_tokens=protocol["steps"] * protocol["batch"]
                * protocol["config"]["context"] * protocol["accumulation"],
                data_signature=protocol["data"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [summarize(path) for path in args.runs]
    if any(row["data_signature"] != rows[0]["data_signature"] for row in rows):
        raise ValueError("Different corpus or tokenizer fingerprints")
    if len({row["training_target_tokens"] for row in rows}) != 1:
        raise ValueError("Training target budgets differ")
    selected = min(rows, key=lambda row: row["best_validation_loss"])
    report = dict(selection_metric="fixed Wind validation-window cross-entropy",
                  selected=selected["run"], runs=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"Selected {selected['run']} at step {selected['best_step']}")


if __name__ == "__main__":
    main()
