"""Pre-norm decoder with multi-head SDPA and tied token embeddings."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int = 1024
    context: int = 128
    width: int = 128
    layers: int = 4
    heads: int = 4

    def __post_init__(self):
        if min(vars(self).values()) <= 0 or self.width % self.heads:
            raise ValueError("Positive dimensions and width divisible by heads required")


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.norm1 = nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.out = nn.Linear(config.width, config.width)
        self.norm2 = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(nn.Linear(config.width, 4 * config.width),
                                 nn.GELU(), nn.Linear(4 * config.width, config.width))

    def forward_cached(self, x, past=None):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q, k, v = [t.view(batch, length, self.heads, width // self.heads).transpose(1, 2)
                   for t in (q, k, v)]
        if past is not None:
            if length != 1:
                raise ValueError("Cached decode requires one new token")
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        # One decode query can see all cached keys. Square prefill uses a causal mask.
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=past is None)
        x = x + self.out(attended.transpose(1, 2).contiguous().view(batch, length, width))
        return x + self.mlp(self.norm2(x)), (k, v)

    def forward(self, x):
        return self.forward_cached(x)[0]


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.width)
        self.position = nn.Embedding(config.context, config.width)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.layers)))
        self.norm = nn.LayerNorm(config.width)
        self.head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.head.weight = self.token.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens):
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.config.context:
            raise ValueError("Expected batch of 1..context tokens")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        return self.head(self.norm(self.blocks(self.token(tokens) + self.position(positions))))

    def forward_cached(self, tokens, cache=None):
        offset = 0 if cache is None else cache[0][0].shape[2]
        if not 0 < tokens.shape[1] + offset <= self.config.context:
            raise ValueError("Cached sequence exceeds context")
        positions = torch.arange(offset, offset + tokens.shape[1], device=tokens.device)
        hidden = self.token(tokens) + self.position(positions)
        updated = []
        for i, block in enumerate(self.blocks):
            hidden, entry = block.forward_cached(hidden, None if cache is None else cache[i])
            updated.append(entry)
        return self.head(self.norm(hidden)), updated

    @torch.inference_mode()
    def generate(self, tokens, count=32, temperature=0.7, top_k=20, use_cache=True):
        if count < 0 or temperature < 0 or top_k < 0:
            raise ValueError("Generation parameters must be nonnegative")
        cache = None
        for _ in range(count):
            if use_cache:
                if cache is None or cache[0][0].shape[2] >= self.config.context:
                    logits, cache = self.forward_cached(tokens[:, -self.config.context:])
                else:
                    logits, cache = self.forward_cached(tokens[:, -1:], cache)
                logits = logits[:, -1]
            else:
                logits = self(tokens[:, -self.config.context:])[:, -1]
            if temperature == 0:
                next_id = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    cutoff = logits.topk(min(top_k, logits.shape[-1])).values[:, -1:]
                    logits = logits.masked_fill(logits < cutoff, float("-inf"))
                next_id = torch.multinomial(logits.softmax(-1), 1)
            tokens = torch.cat((tokens, next_id), dim=1)
        return tokens
