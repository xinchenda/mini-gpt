"""Document-aware sampling with split and duplicate checks."""
import hashlib
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def load_corpus(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest.get("train"), list) or not manifest["train"]:
        raise ValueError("manifest.train must be a nonempty list")
    if any(not isinstance(manifest.get(role), str) for role in
           ("test", "tokenizer_file")):
        raise ValueError("test and tokenizer_file must be paths")
    validation = manifest.get("validation")
    if not isinstance(validation, str) and not (
            isinstance(validation, list) and validation
            and all(isinstance(item, str) for item in validation)):
        raise ValueError("validation must be a path or a nonempty list of paths")
    root = manifest_path.parent
    paths = {role: [(root / p).resolve() for p in
                   (manifest[role] if isinstance(manifest[role], list) else [manifest[role]])]
             for role in ("train", "validation", "test")}
    all_paths = sum(paths.values(), [])
    if len(set(all_paths)) != len(all_paths):
        raise ValueError("Document paths overlap between roles")
    hashes = {str(p): digest(p) for p in all_paths}
    if len(set(hashes.values())) != len(hashes):
        raise ValueError("Duplicate document content detected")
    tokenizer_path = (root / manifest["tokenizer_file"]).resolve()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    signature = {
        "documents": {role: [hashes[str(p)] for p in group] for role, group in paths.items()},
        "tokenizer": digest(tokenizer_path),
    }
    # Test text is fingerprinted here but never tokenized or scored during training.
    if manifest.get("format") == "packed_uint16_le":
        import numpy as np
        documents = {role: [np.memmap(p, dtype="<u2", mode="r") for p in paths[role]]
                     for role in ("train", "validation")}
        return documents, tokenizer, tokenizer_path, signature
    if manifest.get("format") not in (None, "text"):
        raise ValueError("Unsupported corpus format")
    documents = {role: [torch.tensor(tokenizer.encode(p.read_text(encoding="utf-8")).ids)
                        for p in paths[role]] for role in ("train", "validation")}
    return documents, tokenizer, tokenizer_path, signature


def sample_batch(documents, batch, context, generator, document_mixture=0.0):
    counts = torch.tensor([max(0, len(doc) - context) for doc in documents], dtype=torch.float64)
    if counts.sum() == 0:
        raise ValueError("No document is longer than context")
    if not 0 <= document_mixture <= 1:
        raise ValueError("document_mixture must lie in [0, 1]")
    weights = counts
    if document_mixture:
        valid = (counts > 0).to(counts.dtype)
        weights = ((1 - document_mixture) * counts / counts.sum()
                   + document_mixture * valid / valid.sum())
    chosen = torch.multinomial(weights, batch, replacement=True, generator=generator)
    windows = []
    for index in chosen.tolist():
        doc = documents[index]
        start = torch.randint(len(doc) - context, (1,), generator=generator).item()
        windows.append(doc[start:start + context + 1])
    if isinstance(windows[0], torch.Tensor):
        windows = torch.stack(windows)
    else:
        import numpy as np
        windows = torch.from_numpy(np.array(windows, dtype=np.int64))
    return windows[:, :-1], windows[:, 1:]
