"""Data integrity checks through the manifest entry point."""
import json
from pathlib import Path
import tempfile
import unittest
from tokenizers import Tokenizer, models
from minigpt_v2.data import load_corpus
from minigpt_v2.normalize_corpus import normalize


class DataIntegrity(unittest.TestCase):
    def test_line_unwrapping(self):
        text = """The old man
opened the door.

A new paragraph."""
        expected = """The old man opened the door.

A new paragraph."""
        self.assertEqual(normalize(text), expected)
        self.assertEqual(normalize(expected), expected)
        self.assertEqual(normalize(text, preserve_verse=True), text)

    def test_overlap_and_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("a", "b", "c"):
                (root / name).write_text(name * 100)
            tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
            tokenizer.save(str(root / "tokenizer.json"))
            manifest = dict(train=["a"], validation="b", test="c",
                            tokenizer_file="tokenizer.json")
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            documents, _, _, signature = load_corpus(path)
            self.assertEqual(len(signature["documents"]["train"]), 1)
            self.assertNotIn("test", documents)
            (root / "d").write_text("d" * 100)
            manifest["validation"] = ["b", "d"]
            path.write_text(json.dumps(manifest))
            multiple, _, _, _ = load_corpus(path)
            self.assertEqual(len(multiple["validation"]), 2)
            manifest["validation"] = "b"
            path.write_text(json.dumps(manifest))
            (root / "b").write_text("a" * 100)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_corpus(path)
            manifest["validation"] = "a"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_corpus(path)
