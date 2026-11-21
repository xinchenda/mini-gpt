"""Exercise HTTP entry points using an ephemeral local port."""
from http.server import ThreadingHTTPServer
import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from tokenizers import Tokenizer, models, pre_tokenizers
from minigpt_v2.model import Config, GPT
from minigpt_v2.serve import handler_type


class Service(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2},
                                               unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        model = GPT(Config(vocab_size=3, context=8, width=8, layers=1, heads=2)).eval()
        cls.model = model
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_type(model, tokenizer))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_health_and_generation(self):
        with urlopen(self.url + "/health") as result:
            self.assertEqual(json.load(result)["status"], "ready")
        request = Request(self.url + "/generate",
                          data=json.dumps(dict(prompt="hello", max_new_tokens=4)).encode())
        with urlopen(request) as result:
            self.assertIsInstance(json.load(result)["text"], str)

    def test_busy_returns_429(self):
        entered = threading.Event()
        release = threading.Event()
        original = self.model.generate

        def delayed(*args, **kwargs):
            entered.set()
            release.wait(10)
            return original(*args, **kwargs)

        def request():
            body = json.dumps(dict(prompt="hello", max_new_tokens=2)).encode()
            with urlopen(Request(self.url + "/generate", data=body), timeout=15) as response:
                return json.load(response)

        with patch.object(self.model, "generate", side_effect=delayed):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(request)
                try:
                    self.assertTrue(entered.wait(5))
                    with self.assertRaises(HTTPError) as error:
                        request()
                    self.assertEqual(error.exception.code, 429)
                finally:
                    release.set()
                self.assertIn("text", future.result(timeout=15))

    def test_invalid_requests(self):
        for body in (b"[]", b"null", b"not json",
                     b'{"prompt":"hello","max_new_tokens":200}'):
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(self.url + "/generate", data=body))
            self.assertEqual(error.exception.code, 400)
