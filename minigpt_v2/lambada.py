"""Full LAMBADA OpenAI last-word evaluation with explicit context truncation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from urllib.request import urlopen
import torch
from .inference import load_artifact

SOURCE = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"


def make_windows(texts, tokenizer, context):
    groups = {}
    for index, text in enumerate(texts):
        prompt, target = text.rsplit(" ", 1)
        prefix = tokenizer.encode(prompt).ids
        ending = tokenizer.encode(" " + target).ids
        if prefix + ending != tokenizer.encode(text).ids:
            raise ValueError("Prompt/target token boundary requires review")
        if not prefix or not ending:
            raise ValueError("Empty benchmark prompt or target")
        for token in ending:
            window = prefix[-context:]
            groups.setdefault(len(window), []).append((index, window, token))
            prefix.append(token)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact")
    parser.add_argument("--data", type=Path, default=Path("data/benchmarks/lambada_test.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--v1", type=Path)
    args = parser.parse_args()
    if not args.data.exists():
        args.data.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(SOURCE, timeout=30) as response:
            args.data.write_bytes(response.read())
    texts = [json.loads(line)["text"] for line in args.data.read_text().splitlines()]
    if len(texts) != 5153:
        raise ValueError("Expected all 5153 LAMBADA OpenAI examples")
    torch.set_num_threads(2)
    model, tokenizer = load_artifact(args.artifact, args.device)
    old = None
    if args.v1:
        import mlx.core as mx
        from train_mlx import MiniGPT
        from generate import load_checkpoint
        tok, context, _, width, layers = load_checkpoint(
            args.v1, bpe_path=Path(args.artifact) / "tokenizer.json")
        old = MiniGPT(tok.vocab_size, context, width, layers)
        old.load_weights(str(args.v1))
        old.eval()
    scores = {"v2": [0.0] * len(texts)}
    correct = {"v2": [True] * len(texts)}
    if old is not None:
        scores["v1"] = [0.0] * len(texts)
        correct["v1"] = [True] * len(texts)
    # Use the same per-example target tokens; each model crops to its native context.
    old_context = old.context_length if old is not None else model.config.context
    groups = make_windows(texts, tokenizer, max(model.config.context, old_context))
    for rows in groups.values():
        for offset in range(0, len(rows), 32):
            batch = rows[offset:offset + 32]
            x = torch.tensor([row[1] for row in batch], device=args.device)
            y = torch.tensor([row[2] for row in batch], device=args.device)
            with torch.inference_mode():
                logits = model(x[:, -model.config.context:])[:, -1]
                losses = torch.nn.functional.cross_entropy(logits, y, reduction="none").tolist()
                predictions = logits.argmax(-1).tolist()
            for (index, _, target), loss, prediction in zip(batch, losses, predictions):
                scores["v2"][index] += loss
                correct["v2"][index] &= prediction == target
            if old is not None:
                import mlx.nn as nn
                logits = old(mx.array([row[1][-old_context:] for row in batch]))[:, -1]
                losses = nn.losses.cross_entropy(logits, mx.array([row[2] for row in batch]),
                                                 reduction="none").tolist()
                predictions = mx.argmax(logits, axis=-1).tolist()
                for (index, _, target), loss, prediction in zip(batch, losses, predictions):
                    scores["v1"][index] += loss
                    correct["v1"][index] &= prediction == target
    report = dict(
        task="LAMBADA OpenAI", examples=len(texts), source=SOURCE, device=args.device,
        sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        context_tokens=model.config.context, v1_context_tokens=old_context, filter_stop_words=False,
        protocol="Last word including leading space; full target greedy exact match; sliding context",
        task_spec="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/lambada/lambada_openai.yaml",
        models={name: dict(accuracy=sum(correct[name]) / len(texts),
                           correct=sum(correct[name]),
                           last_word_perplexity=math.exp(sum(loss) / len(texts)))
                for name, loss in scores.items()},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
