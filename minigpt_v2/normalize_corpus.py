"""Unwrap typographic line breaks while retaining paragraph and verse boundaries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re


def normalize(text, preserve_verse=False):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if preserve_verse:
        return text.strip()
    paragraphs = re.split(r"\n[ \t]*\n+", text)
    return "\n\n".join(re.sub(r"[ \t]*\n[ \t]*", " ", p).strip()
                        for p in paragraphs if p.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/v2_normalized"))
    parser.add_argument("--control-manifest", type=Path,
                        help="Raw train data with the same normalized validation set")
    args = parser.parse_args()
    source = args.manifest.resolve()
    manifest = json.loads(source.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    output = dict(name="gutenberg_27_paragraph_normalized", train=[],
                  tokenizer_file="tokenizer.json", tokenizer_kind="bpe")
    (args.output / "tokenizer.json").write_bytes(
        (source.parent / manifest["tokenizer_file"]).read_bytes())
    audit = []
    for role in ("train", "validation", "test"):
        entries = manifest[role] if role == "train" else [manifest[role]]
        for entry in entries:
            path = source.parent / entry
            raw = path.read_text(encoding="utf-8")
            cleaned = normalize(raw, preserve_verse="romeo_juliet" in path.name)
            (args.output / path.name).write_text(cleaned, encoding="utf-8")
            if role == "train":
                output["train"].append(path.name)
            else:
                output[role] = path.name
            audit.append(dict(role=role, file=path.name,
                              source_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                              normalized_sha256=hashlib.sha256(cleaned.encode()).hexdigest(),
                              original_newlines=raw.count("\n"), normalized_newlines=cleaned.count("\n")))
    (args.output / "manifest.json").write_text(json.dumps(output, indent=2))
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2))
    if args.control_manifest:
        args.control_manifest.parent.mkdir(parents=True, exist_ok=True)
        base = args.control_manifest.parent.resolve()
        control = dict(
            name="raw_train_normalized_validation_control",
            train=[os.path.relpath((source.parent / entry).resolve(), base)
                   for entry in manifest["train"]],
            validation=os.path.relpath((args.output / output["validation"]).resolve(), base),
            test=os.path.relpath((args.output / output["test"]).resolve(), base),
            tokenizer_file=os.path.relpath((args.output / "tokenizer.json").resolve(), base),
        )
        args.control_manifest.write_text(json.dumps(control, indent=2))
    print(f"Normalized {len(audit)} files, preserving Romeo and Juliet verse lines")


if __name__ == "__main__":
    main()
