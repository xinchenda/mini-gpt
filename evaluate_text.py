"""用全书下一 token loss 和真实续写排序评估已有权重。"""

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from generate import load_checkpoint
from train_mlx import MiniGPT
from train_text_mlx import load_text


def score_windows(model, windows, suffix_length):
    scores = []
    for offset in range(0, len(windows), 16):
        sequences = mx.array(windows[offset:offset + 16])
        logits = model(sequences[:, :-1])[:, -suffix_length:, :]
        targets = sequences[:, -suffix_length:]
        losses = nn.losses.cross_entropy(logits, targets, reduction="none")
        scores.extend(mx.sum(losses, axis=1).tolist())
    return scores


def full_text_loss(model, token_ids, stride):
    context = min(model.context_length, len(token_ids) - 1)
    first_end = context + 1
    total = score_windows(model, [token_ids[:first_end]], context)[0]
    full_ends = range(first_end + stride, len(token_ids) + 1, stride)
    windows = [token_ids[end - context - 1:end] for end in full_ends]
    total += sum(score_windows(model, windows, stride))
    remaining = (len(token_ids) - first_end) % stride
    if remaining:
        total += score_windows(model, [token_ids[-context - 1:]], remaining)[0]
    return total / (len(token_ids) - 1)


def continuation_ranking(model, token_ids):
    context = model.context_length
    suffix_length = 16
    last_start = len(token_ids) - context - 1
    originals = []
    reordered = []
    for index in range(64):
        start = index * last_start // 63
        window = token_ids[start:start + context + 1]
        prompt, suffix = window[:-suffix_length], window[-suffix_length:]
        shuffled = suffix[:4] + suffix[8:12] + suffix[4:8] + suffix[12:]
        if suffix != shuffled:
            originals.append(window)
            reordered.append(prompt + shuffled)
    if not originals:
        raise ValueError("没有可比较的真实续写排序样本")
    correct = score_windows(model, originals, suffix_length)
    altered = score_windows(model, reordered, suffix_length)
    wins = sum(real < fake - 1e-6 for real, fake in zip(correct, altered))
    ties = sum(abs(real - fake) <= 1e-6 for real, fake in zip(correct, altered))
    return {
        "examples": len(correct), "wins": wins, "ties": ties,
        "accuracy": (wins + ties / 2) / len(correct),
        "mean_nll_margin_per_token": (sum(altered) - sum(correct)) / (suffix_length * len(correct)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", type=Path)
    parser.add_argument("--model", nargs=2, metavar=("NAME", "WEIGHTS"), action="append", required=True)
    parser.add_argument("--bpe-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = load_text(args.text)
    stride = 16
    protocol = None
    report = {"text": str(args.text), "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
              "stride": stride, "models": {}}
    for name, weights in args.model:
        if name in report["models"]:
            parser.error(f"模型名重复：{name}")
        path = Path(weights)
        tokenizer, context, step, width, layers = load_checkpoint(path, bpe_path=args.bpe_path)
        signature = (context, {
            key: value for key, value in tokenizer.metadata().items() if key != "tokenizer_file"
        })
        if protocol is None:
            protocol = signature
            token_ids = tokenizer.encode(text, use_unknown=True)
            report.update(context=context, tokens=len(token_ids), scored_tokens=len(token_ids) - 1)
        if signature != protocol:
            parser.error("公平比较要求相同上下文和 tokenizer 文件内容")
        if context < stride or len(token_ids) < context + 64:
            parser.error("评估要求至少 16 token 上下文和 64 个可区分的抽样位置")
        model = MiniGPT(tokenizer.vocab_size, context, width, layers)
        model.load_weights(str(path))
        model.eval()
        result = {
            "weights": weights, "step": step, "width": width, "layers": layers,
            "loss_nat_per_token": full_text_loss(model, token_ids, stride),
            "continuation_ranking": continuation_ranking(model, token_ids),
        }
        report["models"][name] = result
        print(f"{name} 全书 loss {result['loss_nat_per_token']:.4f}，"
              f"真实续写排序 {result['continuation_ranking']['wins']}/{result['continuation_ranking']['examples']}",
              flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
