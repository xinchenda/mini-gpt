"""Cloud preparation invariants that can be verified without a CUDA GPU."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from tokenizers import Tokenizer, models
from minigpt_v2.data import load_corpus, sample_batch
from minigpt_v2.prepare_web import split_for


class CloudData(unittest.TestCase):
    def test_memmap_sampling_and_split(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0, 'x': 1}, unk_token='[UNK]'))
            tokenizer.save(str(root / 'tokenizer.json'))
            for index, role in enumerate(('train', 'validation', 'test')):
                (np.arange(100, dtype='<u2') + index).tofile(root / f'{role}.bin')
            manifest = dict(format='packed_uint16_le', train=['train.bin'],
                            validation='validation.bin', test='test.bin',
                            tokenizer_file='tokenizer.json')
            (root / 'manifest.json').write_text(json.dumps(manifest))
            docs, _, _, signature = load_corpus(root / 'manifest.json')
            self.assertIsInstance(docs['train'][0], np.memmap)
            self.assertNotIn('test', docs)
            x, y = sample_batch(docs['train'], 4, 8, torch.Generator().manual_seed(42))
            self.assertEqual(x.dtype, torch.int64)
            self.assertEqual(x.shape, (4, 8))
            self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))
            self.assertEqual(len(signature['documents']['test']), 1)
        self.assertEqual(split_for('identical content'), split_for('identical content'))
        roles = {split_for(str(i)) for i in range(10000)}
        self.assertEqual(roles, {'train', 'validation', 'test'})
