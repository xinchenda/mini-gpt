"""在相同目标位置比较短上下文、长上下文与替换后的远处前缀。"""

import argparse
import hashlib
import json
from pathlib import Path

from evaluate_text import score_windows
from generate import load_checkpoint
from train_mlx import MiniGPT
from train_text_mlx import load_text


def compare_history(model, tokens):
    suffix_length = 16
    example_count = 1024
    if len(tokens) < 128 + example_count:
        raise ValueError("评估文本过短")
    last_start = len(tokens) - 129
    starts = [index * last_start // (example_count - 1) for index in range(example_count)]
    short = [tokens[start + 64:start + 129] for start in starts]
    short_scores = score_windows(model, short, suffix_length)
    result = {"loss_context64": sum(short_scores) / (example_count * suffix_length)}
    if model.context_length >= 128:
        long = [tokens[start:start + 129] for start in starts]
        donors = long[example_count // 2:] + long[:example_count // 2]
        replaced = [donor[:64] + window[64:] for donor, window in zip(donors, long)]
        long_scores = score_windows(model, long, suffix_length)
        replaced_scores = score_windows(model, replaced, suffix_length)
        result.update(
            loss_context128=sum(long_scores) / (example_count * suffix_length),
            loss_replaced_prefix=sum(replaced_scores) / (example_count * suffix_length),
            real_prefix_wins=sum(real < fake for real, fake in zip(long_scores, replaced_scores)),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", type=Path)
    parser.add_argument("--model", nargs=2, metavar=("NAME", "WEIGHTS"), action="append", required=True)
    parser.add_argument("--bpe-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = load_text(args.text)
    report = {"text": str(args.text), "sha256": hashlib.sha256(text.encode()).hexdigest(),
              "windows": 1024, "targets_per_window": 16, "models": {}}
    for name, weights in args.model:
        if name in report["models"]:
            parser.error(f"模型名重复：{name}")
        tokenizer, context, step, width, layers = load_checkpoint(Path(weights), bpe_path=args.bpe_path)
        if tokenizer.kind != "bpe" or context < 64:
            parser.error("评估要求同一 BPE 词表和至少 64 token 的上下文")
        model = MiniGPT(tokenizer.vocab_size, context, width, layers)
        model.load_weights(weights)
        model.eval()
        result = compare_history(model, tokenizer.encode(text))
        report["models"][name] = {"weights": weights, "step": step, **result}
        print(name, result, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
