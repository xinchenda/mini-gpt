"""Prepare a bounded FineWeb-Edu shard for a single-GPU budget experiment.

Run before renting a GPU. Uses only train documents to learn the tokenizer.
Document IDs deterministically assign validation/test; packed streams use EOS.
"""
import argparse
from array import array
import hashlib
from itertools import batched
import json
from pathlib import Path
import sys
import urllib.request

REVISION = '651a648da38bf545cc5487530dbf59d8168c8de3'
URL = f'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/{REVISION}/sample/10BT/000_00000.parquet'


def split_for(identity):
    bucket = int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16) % 1000
    return 'test' if bucket < 5 else ('validation' if bucket < 10 else 'train')


def main():
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from .data import digest
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/web_cloud'))
    parser.add_argument('--max-train-tokens', type=int, default=300_000_000)
    parser.add_argument('--tokenizer-documents', type=int, default=20000)
    args = parser.parse_args()
    if min(args.max_train_tokens, args.tokenizer_documents) <= 0:
        parser.error('Token and document limits must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'manifest.json').exists():
        raise FileExistsError('Prepared output exists; choose a new directory')
    parquet = args.output / 'source.parquet'
    if not parquet.exists():
        temporary = parquet.with_suffix('.download')
        urllib.request.urlretrieve(URL, temporary)
        temporary.replace(parquet)
    source = pq.ParquetFile(parquet)

    def rows():
        for batch in source.iter_batches(batch_size=128, columns=['id', 'text']):
            for record in batch.to_pylist():
                text = record['text']
                if text and len(text) >= 200:
                    yield record['id'], text

    tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    def tokenizer_data():
        count = 0
        for identity, text in rows():
            if split_for(hashlib.sha256(text.encode()).hexdigest()) != 'train':
                continue
            yield text
            count += 1
            if count >= args.tokenizer_documents:
                return
    tokenizer.train_from_iterator(tokenizer_data(), trainers.BpeTrainer(
        vocab_size=16384, special_tokens=['[UNK]', '[EOS]'],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tokenizer.save(str(args.output / 'tokenizer.json'))
    eos = tokenizer.token_to_id('[EOS]')
    counts = {role: 0 for role in ('train', 'validation', 'test')}
    docs = dict.fromkeys(counts, 0)
    # Exact content duplicate filter also prevents identical texts across splits.
    seen = set()
    duplicates = 0
    streams = {role: (args.output / f'{role}.bin').open('wb') for role in counts}
    try:
        for batch in batched(rows(), 128):
            candidates = []
            for identity, text in batch:
                sha = hashlib.sha256(text.encode()).digest()
                if sha in seen:
                    duplicates += 1
                    continue
                seen.add(sha)
                candidates.append((split_for(sha.hex()), text))
            encoded = tokenizer.encode_batch([text for _, text in candidates])
            for (role, _), encoding in zip(candidates, encoded):
                values = array('H', encoding.ids + [eos])
                if sys.byteorder != 'little':
                    values.byteswap()
                values.tofile(streams[role])
                counts[role] += len(values)
                docs[role] += 1
                if counts['train'] >= args.max_train_tokens:
                    break
            if counts['train'] >= args.max_train_tokens:
                break
    finally:
        for stream in streams.values():
            stream.close()
    if min(counts.values()) < 2048:
        raise ValueError('Not enough tokens in each split; no usable manifest written')
    manifest = dict(format='packed_uint16_le', train=['train.bin'],
                    validation='validation.bin', test='test.bin',
                    tokenizer_file='tokenizer.json')
    audit = dict(source=URL, revision=REVISION, license='ODC-BY 1.0',
                 source_sha256=digest(parquet), token_counts=counts,
                 document_counts=docs, exact_duplicates_removed=duplicates,
                 split='SHA256(SHA256(text).hexdigest()) modulo 1000: test 0..4, validation 5..9',
                 packing='EOS between documents; windows may cross EOS',
                 tokenizer_training_documents=args.tokenizer_documents,
                 limitations=['No approximate deduplication',
                              'No public benchmark contamination audit yet'],
                 sha256={name: digest(args.output / name) for name in
                         ('train.bin', 'validation.bin', 'test.bin', 'tokenizer.json')})
    (args.output / 'audit.json').write_text(json.dumps(audit, indent=2))
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
