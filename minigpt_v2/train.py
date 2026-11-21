"""python -m minigpt_v2.train --help"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import time
import torch
from torch.nn import functional as F
from .data import digest, load_corpus, sample_batch
from .model import Config, GPT


def atomic_save(state, path):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def device_name():
    return "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")


def initialize_weights(model, path, tokenizer_sha, extend_context=False):
    initial = torch.load(path / "best.pt", map_location="cpu", weights_only=True)
    old_config = initial["protocol"]["config"]
    current = asdict(model.config)
    compatible = old_config == current
    if extend_context:
        compatible = (
            old_config["context"] < current["context"]
            and all(old_config[key] == current[key] for key in current if key != "context")
        )
    if not compatible or initial["protocol"]["data"]["tokenizer"] != tokenizer_sha:
        raise ValueError("Initialization architecture or tokenizer mismatch")
    state = dict(initial["model"])
    if extend_context:
        original = state["position.weight"]
        repeats = math.ceil(current["context"] / old_config["context"])
        state["position.weight"] = original.repeat(repeats, 1)[:current["context"]]
    model.load_state_dict(state)
    return digest(path / "best.pt")


def precision_context(device, precision):
    return (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if precision == "bf16" else nullcontext())


@torch.inference_mode()
def validation(model, documents, device, window=0, precision="fp32"):
    generator = torch.Generator().manual_seed(2026)
    losses = []
    for _ in range(8):
        x, y = sample_batch(documents, 8, window or model.config.context, generator)
        x = x[:, -model.config.context:]
        with precision_context(device, precision):
            logits = model(x.to(device))
        suffix = min(16, model.config.context) if window else logits.shape[1]
        losses.append(F.cross_entropy(logits[:, -suffix:].float().flatten(0, 1),
                                      y[:, -suffix:].to(device).flatten()).item())
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/v2-small.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000, help="Total schedule length")
    parser.add_argument("--stop-at", type=int, help="Stop early without changing schedule")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=device_name())
    loading = parser.add_mutually_exclusive_group()
    loading.add_argument("--resume", type=Path)
    loading.add_argument("--init", type=Path, help="Start a new schedule from v2 artifact weights")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--min-learning-rate", type=float, default=0.0001)
    parser.add_argument("--document-mixture", type=float, default=0.0)
    parser.add_argument("--extend-context", action="store_true")
    parser.add_argument("--validation-window", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--max-seconds", type=float, help="Save and exit after this training wall time")
    args = parser.parse_args()
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error("max-seconds must be positive")
    if args.precision == "bf16" and (not args.device.startswith("cuda")
            or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        parser.error("bf16 requires a CUDA GPU with native BF16 support")
    if not 0 <= args.document_mixture <= 1:
        parser.error("document-mixture must lie in [0, 1]")
    if args.extend_context and args.init is None:
        parser.error("--extend-context requires --init")
    if args.validation_window < 0:
        parser.error("validation-window must be nonnegative")
    if min(args.steps, args.batch_size, args.accumulation) <= 0:
        parser.error("steps, batch size and accumulation must be positive")
    if args.stop_at is not None and not 0 < args.stop_at <= args.steps:
        parser.error("stop-at must lie within the schedule")
    if not 0 <= args.min_learning_rate <= args.learning_rate or args.learning_rate <= 0:
        parser.error("Require 0 <= min-learning-rate <= learning-rate and positive maximum")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "last.pt").exists() and args.resume is None:
        parser.error("Output already contains a run; use a new directory or --resume")
    torch.manual_seed(args.seed)
    documents, tokenizer, tokenizer_path, signature = load_corpus(args.manifest)
    config = Config(vocab_size=tokenizer.get_vocab_size(),
                    **json.loads(Path(args.config).read_text()))
    for role, parts in documents.items():
        if not any(len(part) > config.context for part in parts):
            raise ValueError(f"{role} has no document longer than context")
    model = GPT(config).to(args.device)
    initialization = None
    if args.init:
        initialization = initialize_weights(model, args.init, signature["tokenizer"],
                                            args.extend_context)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    generator = torch.Generator().manual_seed(args.seed)
    protocol = dict(steps=args.steps, batch=args.batch_size, accumulation=args.accumulation,
                    seed=args.seed, config=asdict(config), data=signature,
                    learning_rate=args.learning_rate, min_learning_rate=args.min_learning_rate,
                    initialization=initialization, document_mixture=args.document_mixture,
                    validation_window=args.validation_window, precision=args.precision)
    start, best = 0, float("inf")
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        saved_protocol = dict(saved["protocol"])
        saved_protocol.setdefault("learning_rate", 0.001)
        saved_protocol.setdefault("min_learning_rate", 0.0001)
        saved_protocol.setdefault("initialization", None)
        saved_protocol.setdefault("document_mixture", 0.0)
        saved_protocol.setdefault("validation_window", 0)
        saved_protocol.setdefault("precision", "fp32")
        # Warm-start provenance follows the checkpoint on subsequent exact resumes.
        protocol["initialization"] = saved_protocol["initialization"]
        if saved_protocol != protocol:
            raise ValueError("Resume protocol or data fingerprint changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["sampler"])
        torch.set_rng_state(saved["rng"])
        if args.device.startswith("cuda") and "cuda_rng" in saved:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start, best = saved["step"], saved["best"]
        if start >= args.steps:
            raise ValueError("Checkpoint already completed schedule")
    final = args.stop_at or args.steps
    if final <= start:
        raise ValueError("Requested stop point is not after checkpoint step")
    tokenizer.save(str(args.output / "tokenizer.json"))
    (args.output / "config.json").write_text(json.dumps(asdict(config), indent=2))
    (args.output / "provenance.json").write_text(json.dumps(
        dict(protocol=protocol, torch=torch.__version__, python=platform.python_version(),
             device=args.device, manifest=str(Path(args.manifest).resolve()),
             cuda_version=torch.version.cuda,
             gpu=(torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None)), indent=2))
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"device={args.device} parameters={parameter_count:,}", flush=True)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    for step in range(start + 1, final + 1):
        warmup = min(100, max(1, args.steps // 10))
        factor = min(step / warmup, 1.0)
        if step > warmup:
            floor = args.min_learning_rate / args.learning_rate
            factor = floor + (1 - floor) * (1 + math.cos(
                math.pi * (step - warmup) / max(1, args.steps - warmup))) / 2
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * factor
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for _ in range(args.accumulation):
            x, y = sample_batch(documents["train"], args.batch_size, config.context,
                                generator, args.document_mixture)
            with precision_context(args.device, args.precision):
                logits = model(x.to(args.device))
                loss = F.cross_entropy(logits.flatten(0, 1), y.to(args.device).flatten())
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; checkpoint not updated")
            (loss / args.accumulation).backward()
            loss_total += loss.item() / args.accumulation
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        timed_out = args.max_seconds is not None and time.perf_counter() - began >= args.max_seconds
        if step % 100 == 0 or step == final or timed_out:
            val = validation(model, documents["validation"], args.device, args.validation_window, args.precision)
            improved = val < best
            best = min(best, val)
            state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                         sampler=generator.get_state(), rng=torch.get_rng_state(),
                         step=step, best=best, protocol=protocol)
            if args.device.startswith("cuda"):
                state["cuda_rng"] = torch.cuda.get_rng_state_all()
            atomic_save(state, args.output / "last.pt")
            if improved:
                atomic_save(state, args.output / "best.pt")
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - began
            row = dict(step=step, train_loss=loss_total, validation_loss=val,
                       elapsed_seconds=elapsed, precision=args.precision,
                       effective_training_tokens=step * args.batch_size * config.context * args.accumulation,
                       peak_allocated_bytes=(torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None),
                       gradient_norm=float(norm), lr=optimizer.param_groups[0]["lr"],
                       tokens_per_second=(step - start) * args.batch_size * config.context
                       * args.accumulation / elapsed)
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if timed_out:
            print("Training time limit reached; checkpoint saved. Instance remains billable until stopped.", flush=True)
            break


if __name__ == "__main__":
    main()
