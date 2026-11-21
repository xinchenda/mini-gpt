"""Build a paragraph-books + official WikiText-2 train corpus with separate validation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--books-manifest", type=Path, default=Path("data/v2_normalized/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("data/v2_mixed"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audit = {}
    for split in ("train", "valid", "test"):
        url = ("https://raw.githubusercontent.com/pytorch/examples/acc295dc7b90714f1bf47f06004fc19a7fe235c4/"
               f"word_language_model/data/wikitext-2/{split}.txt")
        path = args.output / f"wiki_{split}.txt"
        with urlopen(url, timeout=30) as response:
            data = response.read()
        if len(data) < 100000:
            raise ValueError("Unexpected WikiText file size")
        path.write_bytes(data)
        audit[split] = dict(source=url, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    manifest = json.loads(args.books_manifest.read_text())
    base = args.output.resolve()
    root = args.books_manifest.parent.resolve()
    mixed = dict(
        name="paragraph_books_plus_wikitext2_train",
        train=[os.path.relpath(root / path, base) for path in manifest["train"]] + ["wiki_train.txt"],
        validation=[os.path.relpath(root / manifest["validation"], base), "wiki_valid.txt"],
        test="wiki_test.txt",
        tokenizer_file=os.path.relpath(root / manifest["tokenizer_file"], base),
    )
    (args.output / "manifest.json").write_text(json.dumps(mixed, indent=2))
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2))
    print("Prepared WikiText train, validation and test with distinct roles")


if __name__ == "__main__":
    main()
