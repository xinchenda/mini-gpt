# Third-Party Notices

MiniGPT's original source code and the model artifacts produced by this
project are distributed under the [MIT License](LICENSE). Third-party
software, datasets, and source materials retain their own licenses.

## Training data

- **FineWeb-Edu**: the cloud model uses a fixed revision of the
  `HuggingFaceFW/fineweb-edu` dataset. The dataset is licensed under
  [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/). Its source URL,
  revision, SHA256, split policy, and prepared token counts are recorded in
  `data/web_cloud/audit.json`. Raw text and tokenized binary files are not
  redistributed in this repository.
- **Project Gutenberg**: the earlier local models use English literary texts
  obtained from Project Gutenberg. Book identifiers, source URLs, text
  boundaries, and content fingerprints are retained in `data/legacy_sources.json`
  and `data/gutenberg_v2/sources.json`. Book text is not redistributed in this
  repository. Project Gutenberg's name and terms remain subject to its own
  policies.
- **WikiText-2**: earlier local experiments reference the official WikiText-2
  dataset. Raw dataset files are not redistributed in this repository.

## Software and ideas reviewed

Runtime dependencies include PyTorch, Hugging Face Tokenizers, NumPy,
Matplotlib, and their transitive dependencies. They are installed from their
upstream packages and retain their respective licenses.

The implementation was developed with references to nanoGPT, nanochat, LitGPT,
and SmolLM. The reviewed revisions, licenses, differences, and adopted design
choices are documented in [docs/OPEN_SOURCE_REVIEW.md](docs/OPEN_SOURCE_REVIEW.md).
Those projects retain their own copyright and license terms.
