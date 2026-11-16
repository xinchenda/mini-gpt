"""加载已训练的权重，对比贪心解码和温度采样。"""

import argparse
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from tokenization import CharacterTokenizer, restore_tokenizer
from train_mlx import MiniGPT
from train_text_mlx import load_text


def load_checkpoint(weights, text=None, bpe_path=None):
    """优先从存档恢复词表；旧存档可用训练文本作为回退。"""
    _, metadata = mx.load(str(weights), return_metadata=True)
    if metadata.get("format_version") in {"1", "2"} and metadata.get("tokenizer"):
        try:
            context_length = int(metadata["context_length"])
            total_steps = int(metadata.get("total_steps", 0))
            width = int(metadata.get("width", 32))
            layers = int(metadata.get("layers", 1))
        except (KeyError, TypeError, ValueError):
            raise ValueError("存档元数据格式无效")
        if metadata.get("model") != "MiniGPT":
            raise ValueError(f"不支持的模型类型：{metadata.get('model')!r}")
        tokenizer = restore_tokenizer(metadata, bpe_path)
        return tokenizer, context_length, total_steps, width, layers

    if text is None:
        raise ValueError("这个权重文件没有词表元数据，请用 --text 提供训练文本")
    source = load_text(text)
    tokenizer = CharacterTokenizer(sorted(set(source[: int(len(source) * 0.9)])))
    return tokenizer, 32, 0, 32, 1


def greedy(model, ids, count):
    tokens = list(ids)
    for _ in range(count):
        logits = model(mx.array(tokens[-model.context_length:])[None, :])[0, -1]
        tokens.append(int(mx.argmax(logits).item()))
    return tokens


def sample(model, ids, count, temperature, top_k):
    tokens = list(ids)
    for _ in range(count):
        logits = model(mx.array(tokens[-model.context_length:])[None, :])[0, -1] / temperature
        if top_k is not None:
            threshold = mx.sort(logits)[-top_k]
            logits = mx.where(logits < threshold, -float("inf"), logits)
        tokens.append(int(mx.random.categorical(logits).item()))
    return tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path, help="safetensors 权重文件")
    parser.add_argument("--text", type=Path, help="旧权重缺少元数据时使用的训练文本")
    parser.add_argument("--bpe-path", type=Path, help="覆盖 checkpoint 中记录的 BPE tokenizer 路径")
    parser.add_argument("--prompt", required=True, help="续写前缀")
    parser.add_argument("--count", type=int, default=60, help="续写 token 数")
    parser.add_argument("--temperature", type=float, nargs="+", default=[0.8], help="采样温度，可给多个")
    parser.add_argument("--top-k", type=int, help="只保留分数最高的 k 个 token")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if any(temperature <= 0 for temperature in args.temperature):
        parser.error("--temperature 必须大于 0")
    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k 必须大于 0")

    try:
        tokenizer, context_length, total_steps, width, layers = load_checkpoint(
            args.weights, args.text, args.bpe_path
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    try:
        ids = tokenizer.encode(args.prompt)
    except ValueError as error:
        parser.error(str(error))

    mx.random.seed(args.seed)
    model = MiniGPT(
        tokenizer.vocab_size,
        context_length=context_length,
        width=width,
        layers=layers,
    )
    model.load_weights(str(args.weights))
    parameter_count = sum(param.size for _, param in tree_flatten(model.parameters()))
    print(f"模型 MiniGPT，tokenizer {tokenizer.kind}，宽度 {width}，层数 {layers}，"
          f"参数量 {parameter_count:,}，上下文长度 {context_length}，"
          f"训练步数 {total_steps}，词表 {tokenizer.vocab_size}")

    def show(label, tokens):
        suffix = tokenizer.decode(tokens[len(ids):])
        print(f"{label} {args.prompt!r} → {suffix!r}")

    show("贪心    ", greedy(model, ids, args.count))
    for temperature in args.temperature:
        show(f"温度 {temperature:<4}", sample(model, ids, args.count, temperature, args.top_k))


if __name__ == "__main__":
    main()
