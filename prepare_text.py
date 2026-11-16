"""从 Project Gutenberg 原文生成以指定正文句子开头的 UTF-8 语料。"""

import argparse
from pathlib import Path


def prepare_text(raw_text, start_marker):
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    if "*** START OF THE PROJECT GUTENBERG EBOOK" in text:
        text = text.split("*** START OF THE PROJECT GUTENBERG EBOOK", 1)[1]
        text = text.split("\n", 1)[-1]
    text = text.split("*** END OF THE PROJECT GUTENBERG EBOOK", 1)[0]
    start = text.find(start_marker)
    if start < 0:
        raise ValueError(f"找不到正文起点：{start_marker!r}")
    return text[start:].strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start-marker", required=True, help="正文第一句")
    args = parser.parse_args()

    try:
        raw_text = args.input.read_text(encoding="utf-8-sig")
        text = prepare_text(raw_text, args.start_marker)
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(f"已写入 {args.output}，共 {len(text):,} 个字符")


if __name__ == "__main__":
    main()
