"""Portable artifact loading, generation and complete-document scoring."""
import argparse
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from .model import Config, GPT
from .data import digest


def load_artifact(directory, device="cpu"):
    directory = Path(directory)
    checksums = directory / "sha256.json"
    if checksums.exists():
        expected = json.loads(checksums.read_text())
        required = {"best.pt", "config.json", "tokenizer.json", "provenance.json"}
        if set(expected) != required:
            raise ValueError("Invalid artifact checksum manifest")
        for name, checksum in expected.items():
            if digest(directory / name) != checksum:
                raise ValueError(f"Artifact checksum mismatch: {name}")
    model = GPT(Config(**json.loads((directory / "config.json").read_text())))
    saved = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    expected = saved["protocol"]["data"]["tokenizer"]
    if digest(directory / "tokenizer.json") != expected:
        raise ValueError("Tokenizer fingerprint mismatch")
    if saved["protocol"]["config"] != vars(model.config):
        raise ValueError("Model configuration differs from checkpoint")
    model.load_state_dict(saved["model"])
    model.to(device).eval()
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    return model, tokenizer


def scoring_windows(ids, context, stride=None):
    if len(ids) < 2:
        raise ValueError("At least two tokens required")
    stride = min(16, context) if stride is None else stride
    if not 0 < stride <= context:
        raise ValueError("stride must lie within context")
    first_end = min(len(ids), context + 1)
    yield ids[:first_end], first_end - 1
    previous = first_end
    while previous < len(ids):
        end = min(previous + stride, len(ids))
        yield ids[max(0, end - context - 1):end], end - previous
        previous = end


@torch.inference_mode()
def score(model, ids):
    total, count = 0.0, 0
    context = model.config.context
    for sequence, suffix in scoring_windows(ids, context):
        x = torch.tensor([sequence[:-1]], device=next(model.parameters()).device)
        y = torch.tensor(sequence[-suffix:], device=x.device)
        total += F.cross_entropy(model(x)[0, -suffix:], y, reduction="sum").item()
        count += suffix
    loss = total / count
    return dict(tokens=count, loss=loss, perplexity=math.exp(loss),
                stride=min(16, context), protocol="strided_sliding_windows_v1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact")
    parser.add_argument("--prompt", default="The old man opened the door and")
    parser.add_argument("--text", type=Path)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model, tokenizer = load_artifact(args.artifact, args.device)
    if args.text:
        print(json.dumps(score(model, tokenizer.encode(args.text.read_text()).ids), indent=2))
    else:
        tokens = tokenizer.encode(args.prompt).ids
        if not tokens:
            parser.error("Prompt must encode to at least one token")
        result = model.generate(torch.tensor([tokens], device=args.device), args.count,
                                temperature=args.temperature, top_k=args.top_k)
        print(tokenizer.decode(result[0].tolist()))


if __name__ == "__main__":
    main()
