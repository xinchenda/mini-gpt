"""训练和生成共用的字符级与 Byte-Level BPE tokenizer。"""

import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer


class CharacterTokenizer:
    kind = "character"

    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.char_to_id = {char: index for index, char in enumerate(alphabet)}
        self.unknown_id = len(alphabet)
        self.vocab_size = len(alphabet) + 1

    def encode(self, text, use_unknown=False):
        if use_unknown:
            return [self.char_to_id.get(char, self.unknown_id) for char in text]
        try:
            return [self.char_to_id[char] for char in text]
        except KeyError as error:
            raise ValueError(f"字符不在词表中：{error.args[0]!r}") from error

    def decode(self, token_ids):
        return "".join(
            self.alphabet[token] if token < self.unknown_id else "\ufffd"
            for token in token_ids
        )

    def tokens(self, token_ids):
        return [
            self.alphabet[token] if token < self.unknown_id else "<unknown>"
            for token in token_ids
        ]

    def unknown_count(self, text):
        return sum(char not in self.char_to_id for char in text)

    def unknown_types(self, text):
        return sorted(set(text) - self.char_to_id.keys())

    def metadata(self):
        return {
            "tokenizer": self.kind,
            "tokenizer_file": "",
            "vocab": json.dumps(self.alphabet, ensure_ascii=False),
            "vocab_size": str(self.vocab_size),
        }


class BPETokenizer:
    kind = "bpe"

    def __init__(self, path):
        self.path = Path(path)
        self.tokenizer = Tokenizer.from_file(str(self.path))
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text, use_unknown=False):
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids):
        return self.tokenizer.decode([int(token) for token in token_ids])

    def tokens(self, token_ids):
        vocab = self.tokenizer.get_vocab()
        inverse_vocab = {index: token for token, index in vocab.items()}
        return [inverse_vocab[int(token)] for token in token_ids]

    def unknown_count(self, text):
        return 0

    def unknown_types(self, text):
        return []

    def metadata(self):
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        return {
            "tokenizer": self.kind,
            "tokenizer_file": str(self.path),
            "tokenizer_sha256": digest,
            "vocab": "",
            "vocab_size": str(self.vocab_size),
        }


def build_tokenizer(kind, train_text, bpe_path=None):
    if kind == "character":
        return CharacterTokenizer(sorted(set(train_text)))
    if kind == "bpe":
        if bpe_path is None:
            raise ValueError("使用 BPE 时必须提供 --bpe-path")
        return BPETokenizer(bpe_path)
    raise ValueError(f"不支持的 tokenizer：{kind!r}")


def restore_tokenizer(metadata, bpe_path=None):
    kind = metadata.get("tokenizer")
    if kind == "character":
        try:
            alphabet = json.loads(metadata["vocab"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("存档中的字符词表无效") from error
        return CharacterTokenizer(alphabet)
    if kind == "bpe":
        path = bpe_path or metadata.get("tokenizer_file")
        if not path:
            raise ValueError("存档没有记录 BPE tokenizer 文件")
        tokenizer = BPETokenizer(path)
        expected_sha256 = metadata.get("tokenizer_sha256")
        if expected_sha256 and tokenizer.metadata()["tokenizer_sha256"] != expected_sha256:
            raise ValueError(f"BPE tokenizer 与存档的 SHA256 不一致：{path}")
        return tokenizer
    raise ValueError(f"存档中的 tokenizer 无效：{kind!r}")
