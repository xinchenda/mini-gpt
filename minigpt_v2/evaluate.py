"""Compare released MLX v1 and PyTorch v2 at identical target positions."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from .inference import load_artifact, scoring_windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact")
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--regression-text", type=Path, required=True)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fresh-holdout", action="store_true",
                        help="Label the second document as the newly reserved final test")
    parser.add_argument("--native-context", action="store_true",
                        help="Score each version at its native context with the same target coverage")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    import mlx.core as mx
    import mlx.nn as nn
    from generate import load_checkpoint
    from train_mlx import MiniGPT

    torch.set_num_threads(2)
    model, tokenizer = load_artifact(args.artifact, args.device)
    old_tokenizer, context, _, width, layers = load_checkpoint(
        args.v1, bpe_path=Path(args.artifact) / "tokenizer.json")
    if context != model.config.context and not args.native_context:
        raise ValueError("Context must match for this protocol")
    old = MiniGPT(old_tokenizer.vocab_size, context, width, layers)
    old.load_weights(str(args.v1))
    old.eval()
    report = dict(
        protocol=("native_context_strided_all_targets" if args.native_context
                  else "same_strided_sliding_windows_v1"), stride=16,
        context_v1=context, context_v2=model.config.context, torch_device=args.device,
        note=("Reserved before scoring; selected candidate versus frozen v1 baseline."
              if args.fresh_holdout else "Previously used holdout is a regression set."),
        models={"v1": str(args.v1), "v2": args.artifact}, documents={})
    for role, path in (("validation", args.validation), ("blind_test" if args.fresh_holdout else "regression", args.regression_text)):
        text = path.read_text(encoding="utf-8")
        ids = tokenizer.encode(text).ids
        v1_sum, v2_sum, count = 0.0, 0.0, 0
        totals = {}
        for name, window_context in (("v1", context), ("v2", model.config.context)):
            windows = list(scoring_windows(ids, window_context, stride=16))
            total = 0.0
            scored = 0
            for offset in range(0, len(windows), 16):
                groups = {}
                for window, suffix in windows[offset:offset + 16]:
                    groups.setdefault((len(window), suffix), []).append(window)
                for (_, suffix), group in groups.items():
                    if name == "v2":
                        values = torch.tensor(group, device=args.device)
                        with torch.inference_mode():
                            logits = model(values[:, :-1])[:, -suffix:]
                            total += torch.nn.functional.cross_entropy(
                                logits.flatten(0, 1), values[:, -suffix:].flatten(),
                                reduction="sum").item()
                    else:
                        sequences = mx.array(group)
                        losses = nn.losses.cross_entropy(
                            old(sequences[:, :-1])[:, -suffix:], sequences[:, -suffix:],
                            reduction="sum")
                        total += losses.item()
                    scored += len(group) * suffix
            if scored != len(ids) - 1:
                raise ValueError("Target coverage mismatch")
            totals[name] = total
        v1_sum, v2_sum, count = totals["v1"], totals["v2"], len(ids) - 1
        report["documents"][role] = dict(
            file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            targets=count, bytes=len(text.encode()), whitespace_words=len(text.split()),
            v1_bits_per_byte=v1_sum / (len(text.encode()) * math.log(2)),
            v2_bits_per_byte=v2_sum / (len(text.encode()) * math.log(2)),
            v1_word_perplexity=math.exp(v1_sum / len(text.split())),
            v2_word_perplexity=math.exp(v2_sum / len(text.split())),
            v1_loss=v1_sum / count, v2_loss=v2_sum / count,
            v1_perplexity=math.exp(v1_sum / count), v2_perplexity=math.exp(v2_sum / count))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
