"""Primary-source failure must use a validated mirror, not cache an error page."""
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError
from prepare_corpus import prepare_book


class Download(unittest.TestCase):
    def test_validated_fallback(self):
        content = b"Project Gutenberg [ebook #11]\nBody begins here."
        with tempfile.TemporaryDirectory() as tmp:
            with patch("prepare_corpus.urlopen", side_effect=[URLError("503"), BytesIO(content)]):
                body, source = prepare_book([11, "example", "Body begins"], True, Path(tmp))
            self.assertEqual(body.read_text(), "Body begins here.\n")
            self.assertIn("uwaterloo.ca", source["download_url"])
