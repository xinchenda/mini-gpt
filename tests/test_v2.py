"""Correctness regressions for causality, sampling, scoring and artifacts."""
import tempfile
import unittest
from pathlib import Path
import torch
from minigpt_v2.model import Config, GPT
from minigpt_v2.data import sample_batch
from minigpt_v2.inference import score


class Correctness(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        self.model = GPT(Config(vocab_size=32, context=8, width=16, layers=2, heads=4))

    def test_causality(self):
        left = torch.randint(32, (1, 8))
        right = left.clone()
        right[:, 4:] = (right[:, 4:] + 1) % 32
        torch.testing.assert_close(self.model(left)[:, :4], self.model(right)[:, :4])

    def test_sampler_restores(self):
        docs = [torch.arange(20), torch.arange(20, 40)]
        rng = torch.Generator().manual_seed(1)
        state = rng.get_state()
        x, y = sample_batch(docs, 4, 8, rng)
        rng.set_state(state)
        other, _ = sample_batch(docs, 4, 8, rng)
        self.assertTrue(torch.equal(x, other))
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))
        self.assertTrue(all(int(row.max() - row.min()) == 7 for row in x))

    def test_score_counts_all_targets(self):
        result = score(self.model, [i % 32 for i in range(22)])
        self.assertEqual(result["tokens"], 21)
        self.assertTrue(0 < result["perplexity"] < float("inf"))

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            torch.save(self.model.state_dict(), path)
            other = GPT(self.model.config)
            other.load_state_dict(torch.load(path, weights_only=True))
            x = torch.randint(32, (2, 8))
            torch.testing.assert_close(other(x), self.model(x))

    def test_cached_logits(self):
        tokens = torch.randint(32, (2, 8))
        _, cache = self.model.forward_cached(tokens[:, :4])
        for i in range(4, 8):
            actual, cache = self.model.forward_cached(tokens[:, i:i + 1], cache)
            expected = self.model(tokens[:, :i + 1])[:, -1:]
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_generation_cache_overflow(self):
        tokens = torch.randint(32, (1, 4))
        cached = self.model.generate(tokens, 16, temperature=0, use_cache=True)
        plain = self.model.generate(tokens, 16, temperature=0, use_cache=False)
        self.assertTrue(torch.equal(cached, plain))

    def test_document_mixture_excludes_short_books(self):
        docs = [torch.arange(3), torch.arange(20)]
        x, _ = sample_batch(docs, 10, 8, torch.Generator().manual_seed(2), 1.0)
        self.assertEqual(x.shape, (10, 8))
        with self.assertRaises(ValueError):
            sample_batch(docs, 2, 8, torch.Generator(), 1.1)

    def test_context_extension_preserves_short_prefix(self):
        from minigpt_v2.train import initialize_weights
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            config = vars(self.model.config)
            torch.save({"model": self.model.state_dict(),
                        "protocol": {"config": config, "data": {"tokenizer": "test"}}},
                       directory / "best.pt")
            extended = GPT(Config(vocab_size=32, context=32, width=16, layers=2, heads=4))
            initialize_weights(extended, directory, "test", extend_context=True)
            x = torch.randint(32, (2, 8))
            torch.testing.assert_close(extended(x), self.model(x))
            self.assertEqual(extended.position.weight.shape[0], 32)

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            Config(width=15, heads=4)


if __name__ == "__main__":
    unittest.main()
