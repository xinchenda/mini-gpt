"""Regression: CPU checkpoint resume reproduces uninterrupted training exactly."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers


class Resume(unittest.TestCase):
    def test_full_training_entrypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer.train_from_iterator(["alpha beta gamma delta epsilon"] * 10,
                                         trainers.BpeTrainer(vocab_size=32, special_tokens=["[UNK]"]))
            tokenizer.save(str(root / "tokenizer.json"))
            for name, text in (
                ("train", "alpha beta gamma delta epsilon " * 30),
                ("val", "beta gamma delta epsilon alpha " * 25),
                ("test", "gamma delta epsilon alpha beta " * 20)):
                (root / f"{name}.txt").write_text(text)
            manifest = dict(train=["train.txt"], validation="val.txt", test="test.txt",
                            tokenizer_file="tokenizer.json")
            (root / "manifest.json").write_text(json.dumps(manifest))
            (root / "config.json").write_text(json.dumps(
                dict(context=8, width=16, layers=1, heads=2)))
            common = [
                sys.executable, "-m", "minigpt_v2.train",
                "--manifest", str(root / "manifest.json"),
                "--config", str(root / "config.json"), "--steps", "6",
                "--batch-size", "2", "--accumulation", "2", "--device", "cpu",
            ]
            env = dict(os.environ, OMP_NUM_THREADS="1")
            for options in (
                ["--output", str(root / "full")],
                ["--output", str(root / "split"), "--stop-at", "3"],
                ["--output", str(root / "split"), "--resume", str(root / "split/last.pt")],
            ):
                subprocess.run(common + options, env=env, check=True, capture_output=True)
            subprocess.run(common + ["--output", str(root / "timed"),
                           "--max-seconds", "0.000000001"],
                           env=env, check=True, capture_output=True)
            timed = torch.load(root / "timed/last.pt", weights_only=True)
            self.assertEqual(timed["step"], 1)
            subprocess.run(common + ["--output", str(root / "timed"),
                           "--resume", str(root / "timed/last.pt")],
                           env=env, check=True, capture_output=True)
            timed = torch.load(root / "timed/last.pt", weights_only=True)
            full = torch.load(root / "full/last.pt", weights_only=True)
            split = torch.load(root / "split/last.pt", weights_only=True)
            for key, tensor in full["model"].items():
                self.assertTrue(torch.equal(tensor, split["model"][key]), key)
                self.assertTrue(torch.equal(tensor, timed["model"][key]), key)
            self.assertTrue(torch.equal(full["sampler"], split["sampler"]))
            self.assertEqual(full["best"], split["best"])
            subprocess.run(
                [sys.executable, "-m", "minigpt_v2.export", str(root / "full"),
                 str(root / "artifact")], check=True, capture_output=True, env=env)
            generated = subprocess.run(
                [sys.executable, "-m", "minigpt_v2.inference", str(root / "artifact"),
                 "--prompt", "alpha beta", "--count", "3"],
                check=True, capture_output=True, env=env)
            self.assertTrue(generated.stdout.strip())
            subprocess.run(
                common + ["--output", str(root / "warm"),
                          "--init", str(root / "artifact"),
                          "--learning-rate", "0.0003", "--min-learning-rate", "0.00003"],
                check=True, capture_output=True, env=env)
            warm = torch.load(root / "warm/last.pt", weights_only=True)
            self.assertIsNotNone(warm["protocol"]["initialization"])
            self.assertEqual(warm["step"], 6)
            (root / "artifact/tokenizer.json").write_text("{}")
            corrupted = subprocess.run(
                [sys.executable, "-m", "minigpt_v2.inference", str(root / "artifact")],
                capture_output=True, env=env)
            self.assertNotEqual(corrupted.returncode, 0)
            (root / "val.txt").write_text("changed data " * 40)
            failed = subprocess.run(common + ["--output", str(root / "split"),
                                    "--resume", str(root / "split/last.pt")],
                                    env=env, capture_output=True)
            self.assertNotEqual(failed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
