"""Download a newly reserved book and audit exact long-phrase overlap."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.request import urlopen
from .normalize_corpus import normalize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/v2_holdout"))
    parser.add_argument("--ebook-id", type=int, choices=(33, 43, 77, 139), default=33)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()
    books = {
        139: ("The Lost World", "Mr. Hungerton", "lost_world.txt"),
        43: ("The Strange Case of Dr. Jekyll and Mr. Hyde",
             "Mr. Utterson the lawyer", "jekyll_hyde.txt"),
        33: ("The Scarlet Letter", "A throng of bearded men", "scarlet_letter.txt"),
        77: ("The House of the Seven Gables",
             "Halfway down a by-street of one of our New England towns", "seven_gables.txt"),
    }
    title, marker, filename = books[args.ebook_id]
    prefix = "/".join(str(args.ebook_id)[:-1])
    url = (f"https://mirror.csclub.uwaterloo.ca/gutenberg/{prefix}/"
           f"{args.ebook_id}/{args.ebook_id}-0.txt")
    with urlopen(url, timeout=30) as response:
        raw = response.read()
    text = raw.decode("utf-8-sig").replace("\r\n", "\n")
    start = text.index(marker)
    end = text.index("*** END OF THE PROJECT GUTENBERG EBOOK")
    body = text[start:end].strip()
    if args.normalize:
        body = normalize(body)
    words = re.findall(r"\w+", body.lower())
    spans = {tuple(words[i:i + 40]) for i in range(len(words) - 39)}
    manifest = json.loads(args.manifest.read_text())
    matches = {}
    for entry in manifest["train"]:
        content = (args.manifest.parent / entry).read_text(encoding="utf-8")
        train_words = re.findall(r"\w+", content.lower())
        hits = sum(tuple(train_words[i:i + 40]) in spans for i in range(len(train_words) - 39))
        if hits:
            matches[entry] = hits
    if matches:
        raise ValueError(f"Exact 40-word overlap requires review: {matches}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / filename).write_text(body, encoding="utf-8")
    report = dict(source=url, title=title, gutenberg_id=args.ebook_id, normalized=args.normalize,
                  raw_sha256=hashlib.sha256(raw).hexdigest(),
                  body_sha256=hashlib.sha256(body.encode()).hexdigest(),
                  start_marker=marker, characters=len(body),
                  overlap_40_word_windows=matches,
                  caveat="Exact normalized 40-word check; no semantic or shorter overlap detection")
    (args.output / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
