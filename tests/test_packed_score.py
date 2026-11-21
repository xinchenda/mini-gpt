import unittest
import torch
from minigpt_v2.model import GPT,Config
from minigpt_v2.inference import score
from minigpt_v2.evaluate_packed import score_batched


class PackedScore(unittest.TestCase):
    def test_batching_matches_existing_scorer(self):
        torch.manual_seed(7)
        model=GPT(Config(vocab_size=11,context=4,width=8,layers=1,heads=2)).eval()
        ids=[i%11 for i in range(19)]
        reference=score(model,ids)
        for batch in (1,3,8):
            actual=score_batched(model,ids,batch_size=batch,stride=4)
            self.assertEqual(actual['targets'],18)
            self.assertAlmostEqual(actual['loss'],reference['loss'],places=6)
