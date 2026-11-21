"""Measure local greedy decode with and without KV cache."""
import argparse
import json
from statistics import median
import time
import torch
from .inference import load_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact")
    parser.add_argument("--count", type=int, default=24)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("count must be positive")
    torch.set_num_threads(2)
    model, tokenizer = load_artifact(args.artifact)
    prompt = torch.tensor([tokenizer.encode("The old man opened the door and").ids])
    report = dict(device="cpu", threads=2, generated_tokens=args.count,
                  protocol="5 runs median; 1 warmup per mode")
    report["parameters"] = sum(p.numel() for p in model.parameters())
    report["context_tokens"] = model.config.context
    report["prompt_tokens"] = prompt.shape[1]
    report["cache_overflow_rebuilds_expected"] = max(
        0, prompt.shape[1] + args.count - model.config.context)
    outputs = []
    for cache in (False, True):
        model.generate(prompt, args.count, temperature=0, use_cache=cache)
        times = []
        for _ in range(5):
            start = time.perf_counter()
            out = model.generate(prompt, args.count, temperature=0, use_cache=cache)
            times.append(time.perf_counter() - start)
        outputs.append(out)
        report["cached" if cache else "uncached"] = dict(
            median_seconds=median(times), tokens_per_second=args.count / median(times))
    report["identical_greedy_tokens"] = torch.equal(*outputs)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
