"""用本地 UTF-8 文本训练字符级 MiniGPT。"""

import argparse
import csv
import hashlib
import json
import random
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from tokenization import build_tokenizer, restore_tokenizer
from train_mlx import MiniGPT, complete, next_token_loss


EXPERIMENT_FIELDS = [
    "timestamp",
    "run_name",
    "source",
    "model",
    "width",
    "layers",
    "context_length",
    "vocab_size",
    "learning_rate",
    "warmup_steps",
    "run_steps",
    "start_step",
    "end_step",
    "initial_validation_loss",
    "best_validation_loss",
    "best_step",
    "elapsed_seconds",
    "steps_per_second",
    "tokens_per_second",
    "peak_memory_mb",
    "final_checkpoint",
    "best_checkpoint",
]

METRIC_FIELDS = [
    "timestamp",
    "run_name",
    "width",
    "layers",
    "learning_rate",
    "warmup_steps",
    "step",
    "training_batch_loss",
    "validation_loss",
    "current_learning_rate",
    "is_best",
]


def make_batch(token_ids, starts, window_size):
    # 每个窗口比上下文多一个 token，最后一个 token 提供最后一道题的答案。
    return mx.array([token_ids[start:start + window_size] for start in starts])


def load_text(path):
    """读入 UTF-8 文本，并去掉可能残留的古腾堡页眉页脚。"""
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    if "*** START OF THE PROJECT GUTENBERG EBOOK" in text:
        text = text.split("*** START OF THE PROJECT GUTENBERG EBOOK", 1)[1]
        text = text.split("\n", 1)[-1].split("*** END OF THE PROJECT GUTENBERG EBOOK", 1)[0]
    return text.strip()


def append_csv(path, fieldnames, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def append_experiment(path, row):
    append_csv(path, EXPERIMENT_FIELDS, row)


def append_metric(path, row):
    append_csv(path, METRIC_FIELDS, row)


def load_manifest(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {"name", "train", "validation", "test"}
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"manifest 缺少字段：{sorted(missing)}")
    if not isinstance(manifest["train"], list) or not manifest["train"]:
        raise ValueError("manifest 的 train 必须是非空列表")
    base = path.parent
    return {
        "name": manifest["name"],
        "train": [(base / item).resolve() for item in manifest["train"]],
        "validation": (base / manifest["validation"]).resolve(),
        "test": (base / manifest["test"]).resolve(),
        "tokenizer_kind": manifest.get("tokenizer_kind"),
        "tokenizer_file": (base / manifest["tokenizer_file"]).resolve()
        if manifest.get("tokenizer_file") else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", type=Path, nargs="?", help="本地 UTF-8 文本路径")
    parser.add_argument("--train-text", type=Path, action="append", default=[],
                        help="追加训练文本，可重复传入")
    parser.add_argument("--validation-text", type=Path, help="独立的验证文本，使用训练文本的全部内容训练")
    parser.add_argument("--manifest", type=Path, help="包含训练、验证和测试角色的 JSON 清单")
    parser.add_argument("--tokenizer", choices=("character", "bpe"),
                        help="tokenizer 类型，新训练默认 character")
    parser.add_argument("--bpe-path", type=Path, help="已训练的 Byte-Level BPE tokenizer JSON")
    parser.add_argument("--steps", type=int, default=300, help="训练更新次数")
    parser.add_argument("--seed", type=int, default=42, help="模型初始化和采样随机种子")
    parser.add_argument("--prompt", help="用同一个前缀对比训练前后的续写")
    parser.add_argument("--save", type=Path, help="训练结束后把权重保存到这个 safetensors 文件")
    parser.add_argument("--best-save", type=Path, help="验证 loss 创新低时保存最佳权重")
    parser.add_argument("--checkpoint-every", type=int,
                        help="每隔多少步保存一次可续训 checkpoint")
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("experiments/checkpoints/periodic"))
    parser.add_argument("--load", type=Path, help="从 safetensors 文件加载权重，可配合 --steps 0 只做续写")
    parser.add_argument("--width", type=int, default=32, help="模型的隐藏向量宽度")
    parser.add_argument("--layers", type=int, default=1, help="Transformer block 层数")
    parser.add_argument("--context-length", type=int, default=32, help="每个训练窗口的 token 数")
    parser.add_argument("--batch-size", type=int, help="每步抽取的窗口数，默认 16；恢复训练沿用存档")
    parser.add_argument("--learning-rate", type=float, default=0.003, help="最高学习率")
    parser.add_argument("--min-learning-rate", type=float, default=0.0003,
                        help="余弦衰减的最终学习率")
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant",
                        help="warmup 后的学习率调度")
    parser.add_argument("--clip-grad-norm", type=float,
                        help="全局梯度范数上限，不设置则不裁剪")
    parser.add_argument("--warmup-steps", type=int, default=0, help="学习率从 0 线性升至最高值的步数")
    parser.add_argument("--run-name", help="实验记录中的名称")
    parser.add_argument("--log-file", type=Path, default=Path("experiments/summary.csv"), help="实验记录 CSV")
    parser.add_argument("--metrics-file", type=Path, default=Path("experiments/metrics.csv"), help="逐检查点曲线 CSV")
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps 必须大于或等于 0")
    if args.seed < 0:
        parser.error("--seed 必须大于或等于 0")
    if args.width <= 0:
        parser.error("--width 必须大于 0")
    if args.layers <= 0:
        parser.error("--layers 必须大于 0")
    if args.context_length <= 0:
        parser.error("--context-length 必须大于 0")
    if args.checkpoint_every is not None and args.checkpoint_every <= 0:
        parser.error("--checkpoint-every 必须大于 0")
    if args.manifest is None and args.text is None:
        parser.error("需要位置参数 text 或 --manifest")
    if args.manifest is not None and (args.text is not None or args.train_text):
        parser.error("--manifest 不能与 text/--train-text 同时使用")

    try:
        manifest = load_manifest(args.manifest) if args.manifest is not None else None
        training_paths = manifest["train"] if manifest is not None else [args.text, *args.train_text]
        training_parts = [load_text(path) for path in training_paths]
        text = "\n\n".join(training_parts)
        source_label = " + ".join(path.name for path in training_paths)
        validation_path = (
            args.validation_text
            or (manifest["validation"] if manifest is not None else None)
        )
        external_validation_text = (
            load_text(validation_path) if validation_path is not None else None
        )
        tokenizer_kind = args.tokenizer or (
            manifest["tokenizer_kind"] if manifest is not None else None
        )
        bpe_path = args.bpe_path or (
            manifest["tokenizer_file"] if manifest is not None else None
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    saved_metadata = {}
    if args.load is not None:
        try:
            _, saved_metadata = mx.load(str(args.load), return_metadata=True)
        except (OSError, ValueError) as error:
            parser.error(f"无法读取权重文件：{error}")

    try:
        context_length = int(saved_metadata.get("context_length", args.context_length))
        batch_size = int(saved_metadata.get("batch_size", "16")) if saved_metadata else (
            16 if args.batch_size is None else args.batch_size
        )
        loaded_steps = int(saved_metadata.get("total_steps", 0))
        seed = int(saved_metadata.get("seed", args.seed))
        saved_clip = saved_metadata.get("clip_grad_norm")
        clip_grad_norm = (float(saved_clip) if saved_clip else None) if saved_clip is not None else args.clip_grad_norm
        width = int(saved_metadata.get("width", args.width))
        layers = int(saved_metadata.get("layers", args.layers))
        max_lr = float(saved_metadata.get("learning_rate", args.learning_rate))
        min_lr = float(saved_metadata.get("min_learning_rate", args.min_learning_rate))
        warmup_steps = int(saved_metadata.get("warmup_steps", args.warmup_steps))
        lr_schedule = saved_metadata.get("lr_schedule", args.lr_schedule)
    except ValueError:
        parser.error("存档中的模型配置无效")
    if batch_size <= 0:
        parser.error("--batch-size 必须大于 0")
    if saved_metadata and args.batch_size is not None and args.batch_size != batch_size:
        parser.error("--batch-size 与存档不一致")
    if width <= 0:
        parser.error("存档中的模型宽度无效")
    if layers <= 0:
        parser.error("存档中的层数无效")
    if max_lr <= 0:
        parser.error("学习率必须大于 0")
    if not 0 <= min_lr <= max_lr:
        parser.error("最低学习率必须在 0 到最高学习率之间")
    if lr_schedule not in {"constant", "cosine"}:
        parser.error("学习率调度必须为 constant 或 cosine")
    if clip_grad_norm is not None and clip_grad_norm <= 0:
        parser.error("--clip-grad-norm 必须大于 0")
    if args.steps and args.clip_grad_norm is not None and args.clip_grad_norm != clip_grad_norm:
        parser.error("--clip-grad-norm 与存档不一致")
    if saved_metadata.get("validation_mode") == "external" and validation_path is None:
        parser.error("该存档使用独立验证文本，请提供 --validation-text 或 --manifest")
    # 续训时沿用原实验的总调度长度，避免余弦学习率被重新从头计算。
    try:
        saved_schedule_steps = int(saved_metadata.get("schedule_steps", 0))
    except ValueError:
        parser.error("存档中的 schedule_steps 无效")
    schedule_steps = (
        saved_schedule_steps
        if saved_schedule_steps > 0
        else loaded_steps + args.steps
    )
    if args.steps and schedule_steps < loaded_steps + args.steps:
        parser.error("schedule_steps 小于目标训练步数")
    if args.steps and (not 0 <= warmup_steps <= schedule_steps
                       or (lr_schedule == "cosine" and warmup_steps == schedule_steps)):
        parser.error("warmup 步数与完整调度长度不兼容")
    run_name = args.run_name or (
        f"w{width}_l{layers}_s{args.steps}_"
        f"warm{warmup_steps}_lr{max_lr:g}"
    )
    window_size = context_length + 1
    if external_validation_text is not None:
        split_at = len(text)
        validation_mode = "external"
        validation_source = validation_path.name
    else:
        try:
            split_at = int(saved_metadata.get("split_at", int(len(text) * 0.9)))
        except ValueError:
            parser.error("存档中的 split_at 无效")
        validation_mode = "same_source_90_10"
        validation_source = source_label
    if validation_mode == "same_source_90_10" and not 1 <= split_at < len(text):
        parser.error("训练段和验证段的切分位置无效")
    train_text = text[:split_at]
    validation_text = external_validation_text or text[split_at:]
    data_hashes = {
        "training_sha256": hashlib.sha256(train_text.encode()).hexdigest(),
        "validation_sha256": hashlib.sha256(validation_text.encode()).hexdigest(),
    }
    if args.steps and any(saved_metadata.get(key, value) != value for key, value in data_hashes.items()):
        parser.error("训练或验证语料与存档的 SHA256 不一致")
    if min(len(train_text), len(validation_text)) < window_size:
        parser.error("训练段和验证段都需要至少 33 个字符")

    try:
        if saved_metadata:
            tokenizer = restore_tokenizer(saved_metadata, bpe_path)
            if args.tokenizer is not None and args.tokenizer != tokenizer.kind:
                parser.error("--tokenizer 与存档中的 tokenizer 不一致")
        else:
            tokenizer = build_tokenizer(tokenizer_kind or "character", train_text, bpe_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.prompt is not None:
        if not args.prompt:
            parser.error("--prompt 不能为空")
        try:
            tokenizer.encode(args.prompt)
        except ValueError as error:
            parser.error(str(error))
    train_ids = tokenizer.encode(train_text)
    validation_ids = tokenizer.encode(validation_text, use_unknown=True)
    unknown_validation_chars = tokenizer.unknown_types(validation_text)
    unknown_validation_count = tokenizer.unknown_count(validation_text)

    # 每次评估都用同一组均匀分布在验证段内的窗口。
    last_validation_start = len(validation_ids) - window_size
    validation_starts = sorted({i * last_validation_start // 63 for i in range(64)})
    validation = make_batch(validation_ids, validation_starts, window_size)

    mx.random.seed(seed)
    sampler = random.Random(seed)
    model = MiniGPT(tokenizer.vocab_size, context_length, width, layers)
    decay_steps = max(schedule_steps - warmup_steps, 1)
    if lr_schedule == "cosine":
        after_warmup = optim.cosine_decay(
            max_lr,
            decay_steps,
            min_lr,
        )
        if warmup_steps:
            learning_rate = optim.join_schedules(
                [
                    optim.linear_schedule(0, max_lr, warmup_steps),
                    after_warmup,
                ],
                [warmup_steps],
            )
        else:
            learning_rate = after_warmup
    else:
        after_warmup = optim.linear_schedule(
            max_lr,
            max_lr,
            decay_steps,
        )
        if warmup_steps:
            learning_rate = optim.join_schedules(
                [optim.linear_schedule(0, max_lr, warmup_steps), after_warmup],
                [warmup_steps],
            )
        else:
            learning_rate = max_lr
    optimizer = optim.AdamW(learning_rate=learning_rate)
    loss_and_grad = nn.value_and_grad(model, next_token_loss)

    try:
        best_validation_loss = float(saved_metadata.get("best_validation_loss", "inf"))
        best_step = int(saved_metadata.get("best_step", loaded_steps))
    except ValueError:
        parser.error("存档中的最佳验证记录无效")

    def save_checkpoint(path, total_steps):
        path.parent.mkdir(parents=True, exist_ok=True)
        optimizer_path = path.with_suffix(".optimizer.safetensors")
        metadata = {
            "format_version": "2",
            "model": "MiniGPT",
            "seed": str(seed),
            "width": str(width),
            "layers": str(layers),
            "context_length": str(context_length),
            "batch_size": str(batch_size),
            "split_at": str(split_at),
            "total_steps": str(total_steps),
            "learning_rate": str(max_lr),
            "min_learning_rate": str(min_lr),
            "lr_schedule": lr_schedule,
            "warmup_steps": str(warmup_steps),
            "schedule_steps": str(schedule_steps),
            "clip_grad_norm": str(clip_grad_norm) if clip_grad_norm is not None else "",
            "optimizer_file": optimizer_path.name,
            "sampler_state": json.dumps(sampler.getstate()),
            "best_validation_loss": f"{best_validation_loss:.6f}" if best_validation_loss < float("inf") else "",
            "best_step": str(best_step),
            "training_chars": str(len(train_text)),
            "validation_chars": str(len(validation_text)),
            "training_tokens": str(len(train_ids)),
            "validation_tokens": str(len(validation_ids)),
            "source_text": source_label,
            "validation_text": validation_source,
            "validation_mode": validation_mode,
            **data_hashes,
        }
        metadata.update(tokenizer.metadata())
        mx.save_safetensors(
            str(path),
            dict(tree_flatten(model.parameters())),
            metadata=metadata,
        )
        mx.save_safetensors(
            str(optimizer_path),
            dict(tree_flatten(optimizer.state)),
        )

    def show_completion(label):
        prefix = mx.array(tokenizer.encode(args.prompt))
        tokens = complete(model, prefix, count=60)
        generated = tokens[len(prefix):].tolist()
        suffix = tokenizer.decode(generated)
        print(f"{label}续写 {args.prompt!r} → {suffix!r}", flush=True)

    print(f"设备 {mx.default_device()}，训练 token {len(train_ids):,}，"
          f"验证 token {len(validation_ids):,}")
    print(f"训练文本 {source_label}，验证文本 {validation_source}")
    if manifest is not None:
        print(f"数据清单 {manifest['name']}，最终测试保留 {manifest['test'].name}")
    print(f"验证集未知字符 {unknown_validation_count:,} 个，字符类型 {unknown_validation_chars!r}")
    parameter_count = sum(param.size for _, param in tree_flatten(model.parameters()))
    print(f"模型宽度 {width}，层数 {layers}，参数量 {parameter_count:,}")
    print(f"学习率调度 {lr_schedule}，最高 {max_lr:g}，"
          f"最低 {min_lr:g}，warmup {warmup_steps}")
    print(f"梯度裁剪 {clip_grad_norm if clip_grad_norm is not None else '未启用'}")
    print(f"Tokenizer {tokenizer.kind}，词表 {tokenizer.vocab_size}，"
          f"固定验证窗口 {len(validation_starts)} 个")
    print(f"输入 token 示例 {tokenizer.tokens(train_ids[:context_length])!r}")
    print(f"目标 token 示例 {tokenizer.tokens(train_ids[1:window_size])!r}")
    if args.load is not None:
        model.load_weights(str(args.load))
        print(f"已加载权重 {args.load}", flush=True)
        optimizer_file = saved_metadata.get("optimizer_file")
        if args.steps and not optimizer_file:
            parser.error("该 checkpoint 没有优化器状态，不能严格续训")
        if args.steps and optimizer_file:
            optimizer_path = args.load.parent / optimizer_file
            if not optimizer_path.exists():
                parser.error(f"缺少优化器状态文件：{optimizer_path}")
            optimizer_state = mx.load(str(optimizer_path))
            optimizer.state.clear()
            optimizer.state.update(tree_unflatten(list(optimizer_state.items())))
            mx.eval(optimizer.state)
            print(f"已加载优化器状态 {optimizer_path}", flush=True)
        if args.steps and saved_metadata.get("sampler_state"):
            state = json.loads(saved_metadata["sampler_state"])
            sampler.setstate((state[0], tuple(state[1]), state[2]))
    initial_validation_loss = next_token_loss(model, validation).item()
    if initial_validation_loss < best_validation_loss:
        best_validation_loss = initial_validation_loss
        best_step = loaded_steps
    print(f"第 {loaded_steps} 步  验证 loss {initial_validation_loss:.3f}", flush=True)
    if args.steps:
        append_metric(
            args.metrics_file,
            {
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "run_name": run_name,
                "width": width,
                "layers": layers,
                "learning_rate": max_lr,
                "warmup_steps": warmup_steps,
                "step": loaded_steps,
                "validation_loss": f"{initial_validation_loss:.6f}",
                "is_best": 0,
            },
        )
    if args.prompt is not None and args.load is None:
        show_completion("训练前")

    mx.reset_peak_memory()
    training_started = time.perf_counter()
    gradient_norm_sum = 0.0
    maximum_gradient_norm = 0.0
    clipped_update_count = 0
    for step in range(1, args.steps + 1):
        completed_step = loaded_steps + step
        starts = [sampler.randrange(len(train_ids) - context_length) for _ in range(batch_size)]
        batch = make_batch(train_ids, starts, window_size)
        loss, grads = loss_and_grad(model, batch)
        clip_limit = clip_grad_norm if clip_grad_norm is not None else float("inf")
        grads, gradient_norm = optim.clip_grad_norm(grads, clip_limit)
        gradient_norm_value = float(gradient_norm.item())
        gradient_norm_sum += gradient_norm_value
        maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm_value)
        if clip_grad_norm is not None and gradient_norm_value > clip_grad_norm:
            clipped_update_count += 1
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        if completed_step % 100 == 0 or step == args.steps:
            validation_loss = next_token_loss(model, validation).item()
            current_lr = float(optimizer.learning_rate.item())
            print(f"第 {completed_step} 步  更新前训练批次 loss {loss.item():.3f}  "
                  f"更新后验证 loss {validation_loss:.3f}  "
                  f"学习率 {current_lr:.6f}", flush=True)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_step = completed_step
                if args.best_save is not None:
                    save_checkpoint(args.best_save, completed_step)
                    print(f"            新的最佳验证 loss，已保存第 {completed_step} 步权重", flush=True)
            append_metric(
                args.metrics_file,
                {
                    "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "run_name": run_name,
                    "width": width,
                    "layers": layers,
                    "learning_rate": max_lr,
                    "warmup_steps": warmup_steps,
                    "step": completed_step,
                    "training_batch_loss": f"{loss.item():.6f}",
                    "validation_loss": f"{validation_loss:.6f}",
                    "current_learning_rate": f"{current_lr:.8f}",
                    "is_best": int(validation_loss == best_validation_loss),
                },
            )
        if args.checkpoint_every and completed_step % args.checkpoint_every == 0:
            checkpoint_path = args.checkpoint_dir / f"checkpoint_step_{completed_step}.safetensors"
            save_checkpoint(checkpoint_path, completed_step)
            print(f"            已保存周期 checkpoint {checkpoint_path}", flush=True)

    elapsed = time.perf_counter() - training_started
    processed_tokens = args.steps * batch_size * context_length
    steps_per_second = args.steps / elapsed if elapsed else 0
    tokens_per_second = processed_tokens / elapsed if elapsed else 0
    peak_memory_mb = mx.get_peak_memory() / 1e6
    if args.steps:
        print(f"训练耗时 {elapsed:.3f} 秒，{steps_per_second:.1f} 步/秒，"
              f"{tokens_per_second:,.0f} token/秒，峰值内存 {peak_memory_mb:.1f} MB")
        print(f"梯度范数平均 {gradient_norm_sum / args.steps:.3f}，"
              f"最大 {maximum_gradient_norm:.3f}，裁剪更新 {clipped_update_count}/{args.steps}")

    final_step = loaded_steps + args.steps
    if args.save is not None:
        save_checkpoint(args.save, final_step)
        print(f"已保存权重 {args.save}", flush=True)
    print(f"最佳验证 loss {best_validation_loss:.3f}，出现在第 {best_step} 步", flush=True)
    if args.steps:
        append_experiment(
            args.log_file,
            {
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
                "run_name": run_name,
                "source": source_label,
                "model": "MiniGPT",
                "width": width,
                "layers": layers,
                "context_length": context_length,
                "vocab_size": tokenizer.vocab_size,
                "learning_rate": max_lr,
                "warmup_steps": warmup_steps,
                "run_steps": args.steps,
                "start_step": loaded_steps,
                "end_step": final_step,
                "initial_validation_loss": f"{initial_validation_loss:.6f}",
                "best_validation_loss": f"{best_validation_loss:.6f}",
                "best_step": best_step,
                "elapsed_seconds": f"{elapsed:.3f}",
                "steps_per_second": f"{steps_per_second:.1f}",
                "tokens_per_second": f"{tokens_per_second:.0f}",
                "peak_memory_mb": f"{peak_memory_mb:.1f}",
                "final_checkpoint": str(args.save) if args.save is not None else "",
                "best_checkpoint": str(args.best_save) if args.best_save is not None else "",
            },
        )
        print(f"已记录实验 {args.log_file}", flush=True)
    if args.prompt is not None and (args.steps > 0 or args.load is not None):
        show_completion("加载后" if args.load is not None else "训练后")


if __name__ == "__main__":
    main()
