"""Export inference weights and verify tokenizer provenance."""
import argparse
import json
from pathlib import Path
import shutil
import torch
from .data import digest
from .inference import load_artifact
from .train import atomic_save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Export destination already exists")
    model, _ = load_artifact(args.run)
    state = torch.load(args.run / "best.pt", map_location="cpu", weights_only=True)
    if digest(args.run / "tokenizer.json") != state["protocol"]["data"]["tokenizer"]:
        raise ValueError("Tokenizer fingerprint mismatch")
    args.output.mkdir(parents=True)
    for filename in ("tokenizer.json", "config.json", "provenance.json"):
        shutil.copyfile(args.run / filename, args.output / filename)
    atomic_save(dict(model=model.state_dict(), step=state["step"],
                     protocol=state["protocol"]), args.output / "best.pt")
    manifest = {p.name: digest(p) for p in args.output.iterdir() if p.is_file()}
    (args.output / "sha256.json").write_text(json.dumps(manifest, indent=2))
    load_artifact(args.output)
    print(f"Exported {sum(p.stat().st_size for p in args.output.iterdir()):,} bytes")


if __name__ == "__main__":
    main()
