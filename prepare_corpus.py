"""重建 Gutenberg V2 数据清单，记录文件指纹并检查跨书重复。"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from urllib.request import urlopen

from prepare_text import prepare_text


DATA = Path(__file__).resolve().parent / "data"
SOURCES = DATA / "gutenberg_v2" / "sources.json"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_book(source, download, directory=SOURCES.parent):
    ebook, name, marker = source
    url = f"https://www.gutenberg.org/cache/epub/{ebook}/pg{ebook}.txt"
    raw_path = directory / ("alice.txt" if name == "alice" else f"{name}_raw.txt")
    source_record = raw_path.with_suffix(".source.json")
    if download and not raw_path.exists():
        prefix = "/".join(str(ebook)[:-1])
        mirror = f"https://mirror.csclub.uwaterloo.ca/gutenberg/{prefix}/{ebook}/{ebook}-0.txt"
        errors = []
        for candidate in (url, mirror):
            try:
                with urlopen(candidate, timeout=30) as response:
                    content = response.read()
                decoded = content.decode("utf-8-sig")
                if not re.search(rf"\[ebook\s+#0*{ebook}\]", decoded[:4096], re.IGNORECASE):
                    raise ValueError("Downloaded book ID does not match")
                prepare_text(decoded, marker)
                temporary = raw_path.with_suffix(".download")
                temporary.write_bytes(content)
                temporary.replace(raw_path)
                source_record.write_text(json.dumps(
                    {"download_url": candidate, "raw_sha256": hashlib.sha256(content).hexdigest()},
                    indent=2))
                break
            except (OSError, ValueError) as error:
                errors.append(f"{candidate}: {error}")
        else:
            raise ValueError("All download sources failed: " + "; ".join(errors))
    raw = raw_path.read_text(encoding="utf-8-sig")
    if not re.search(rf"\[ebook\s+#0*{ebook}\]", raw[:4096], re.IGNORECASE):
        raise ValueError(f"{raw_path.name} 与清单中的 Gutenberg ID {ebook} 不符")
    body = prepare_text(raw, marker)
    body_path = directory / f"{name}_body.txt"
    content = body + "\n"
    if body_path.exists() and body_path.read_text(encoding="utf-8") != content:
        raise ValueError(f"{body_path.name} 已存在且正文不同，请保留旧版后另行检查")
    if not body_path.exists():
        body_path.write_text(content, encoding="utf-8")
    downloaded = json.loads(source_record.read_text()) if source_record.exists() else {}
    return body_path, {"ebook": ebook, "url": url, "start_marker": marker, **downloaded}


def audit(manifest, provenance):
    book_hashes = defaultdict(list)
    passages = defaultdict(Counter)
    books = []
    for role in ("train", "validation", "test"):
        paths = manifest[role] if role == "train" else [manifest[role]]
        for path in paths:
            raw = (DATA / path).read_bytes()
            text = raw.decode("utf-8-sig").strip()
            identity = " ".join(re.findall(r"[^\W_]+", text.casefold()))
            book_hashes[hashlib.sha256(identity.encode()).hexdigest()].append(path)
            books.append({
                "role": role, "file": path, "characters": len(text),
                "sha256": hashlib.sha256(raw).hexdigest(), **provenance.get(path, {}),
            })
            for paragraph in re.split(r"\n\s*\n", text):
                words = re.findall(r"[^\W_]+", paragraph.casefold())
                if len(words) >= 40:
                    passages[" ".join(words)][path] += 1

    roles = {book["file"]: book["role"] for book in books}
    repeated = [
        {"words": len(passage.split()), "books": dict(counts), "preview": passage[:100]}
        for passage, counts in passages.items() if len(counts) > 1
    ]
    overlap = [
        group for group in repeated
        if "train" in {roles[path] for path in group["books"]}
        and len({roles[path] for path in group["books"]}) > 1
    ]
    return {
        "method": "整书与至少 40 词的完整段落，忽略大小写、标点与空白后精确匹配",
        "books": books,
        "duplicate_books": [paths for paths in book_hashes.values() if len(paths) > 1],
        "repeated_paragraphs_across_books": repeated,
        "train_eval_overlap": overlap,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="下载缺失的原文")
    args = parser.parse_args()
    config = json.loads(SOURCES.read_text(encoding="utf-8"))
    base = json.loads((SOURCES.parent / config["base_manifest"]).read_text(encoding="utf-8"))
    legacy_sources = json.loads((DATA / "legacy_sources.json").read_text(encoding="utf-8"))
    legacy = [prepare_book(book, args.download, DATA) for book in legacy_sources]
    additions = [prepare_book(book, args.download) for book in config["train"]]
    holdout, holdout_source = prepare_book(config["test"], args.download)
    release_holdout = prepare_book(config["release_test"], args.download)
    provenance = {
        path.relative_to(DATA).as_posix(): source
        for path, source in [*legacy, *additions, (holdout, holdout_source), release_holdout]
    }
    manifest = {
        **base,
        "name": "gutenberg_v2_27books",
        "train": [*base["train"], *(path.relative_to(DATA).as_posix() for path, _ in additions)],
        "test": holdout.relative_to(DATA).as_posix(),
    }
    paths = [*manifest["train"], manifest["validation"], manifest["test"]]
    if len(set(paths)) != len(paths):
        parser.error("同一文件重复出现，请检查 manifest 数据角色")
    report = audit(manifest, provenance)
    write_json(SOURCES.parent / "audit.json", report)
    if report["duplicate_books"] or report["train_eval_overlap"]:
        parser.error("发现整书重复或训练与评估重合，请先核对 audit.json")
    release_report = audit({**manifest, "test": release_holdout[0].relative_to(DATA).as_posix()}, provenance)
    write_json(SOURCES.parent / "release_audit.json", release_report)
    if release_report["duplicate_books"] or release_report["train_eval_overlap"]:
        parser.error("发布测试书与现有语料重合，请先核对 release_audit.json")
    write_json(DATA / "corpus_manifest_27books.json", manifest)
    print(f"已审计 {len(manifest['train'])} 本训练书和两本评估书。"
          f"跨书重复长段落 {len(report['repeated_paragraphs_across_books'])} 组。")


if __name__ == "__main__":
    main()
