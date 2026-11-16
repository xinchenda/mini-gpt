"""在一组 UTF-8 文本上训练 Byte-Level BPE tokenizer。"""

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def read_texts(paths):
    return [path.read_text(encoding="utf-8-sig") for path in paths]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="*", help="训练 tokenizer 的文本")
    parser.add_argument("--manifest", type=Path, help="只使用清单中的 train 文本")
    parser.add_argument("--output", type=Path, default=Path("data/bpe_tokenizer.json"))
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--sample-chars", type=int, default=1000)
    args = parser.parse_args()
    if args.vocab_size < 256:
        parser.error("Byte-Level BPE 的 --vocab-size 至少为 256")
    if args.sample_chars <= 0:
        parser.error("--sample-chars 必须大于 0")
    if args.manifest is None and not args.inputs:
        parser.error("需要输入文本或 --manifest")
    if args.manifest is not None and args.inputs:
        parser.error("--manifest 不能与 inputs 同时使用")

    try:
        if args.manifest is not None:
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            if not isinstance(manifest.get("train"), list) or not manifest["train"]:
                raise ValueError("manifest 的 train 必须是非空列表")
            inputs = [(args.manifest.parent / item).resolve() for item in manifest["train"]]
        else:
            inputs = args.inputs
        texts = read_texts(inputs)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.output))

    sample = texts[0][:args.sample_chars]
    encoding = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoding.ids)
    if decoded != sample:
        raise ValueError("tokenizer 解码结果与原文不一致")

    compression = len(sample) / len(encoding.ids)
    print(f"词表大小 {tokenizer.get_vocab_size():,}")
    print(f"样本字符 {len(sample):,}，BPE token {len(encoding.ids):,}，"
          f"平均每个 token {compression:.2f} 个字符")
    print(f"前 20 个 token {encoding.tokens[:20]}")
    print(f"解码一致性检查通过，已保存 {args.output}")


if __name__ == "__main__":
    main()
