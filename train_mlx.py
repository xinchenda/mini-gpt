"""用 MLX 从随机权重训练字符级 GPT。"""

import argparse

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from train import make_data


class TransformerBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_output = nn.Linear(width, width)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 4 * width),
            nn.GELU(), nn.Linear(4 * width, width),
        )

    def __call__(self, hidden):
        query, key, value = mx.split(self.qkv(self.attention_norm(hidden)), 3, axis=-1)
        # MLX 注意力接收 [批次, 头数, 序列长度, 每头维度]。
        attended = mx.fast.scaled_dot_product_attention(
            query[:, None], key[:, None], value[:, None],
            scale=query.shape[-1] ** -0.5, mask="causal",
        )
        hidden = hidden + self.attention_output(attended[:, 0])
        hidden = hidden + self.feed_forward(hidden)
        return hidden


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, context_length, width=32, layers=1):
        super().__init__()
        self.context_length = context_length
        self.width = width
        self.layers = layers
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(context_length, width)
        self.blocks = [TransformerBlock(width) for _ in range(layers)]
        self.readout = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, vocab_size))

    def __call__(self, tokens):
        positions = mx.arange(tokens.shape[1])
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        for block in self.blocks:
            hidden = block(hidden)
        return self.readout(hidden)


def next_token_loss(model, sequences, suffix_length=None):
    inputs, targets = sequences[:, :-1], sequences[:, 1:]
    logits = model(inputs)
    if suffix_length is not None:
        logits = logits[:, -suffix_length:, :]
        targets = targets[:, -suffix_length:]
    return nn.losses.cross_entropy(logits, targets, reduction="mean")


def complete(model, prefix, count):
    tokens = prefix[None, :]
    for _ in range(count):
        logits = model(tokens[:, -model.context_length:])
        next_token = mx.argmax(logits[:, -1], axis=-1, keepdims=True)
        tokens = mx.concatenate((tokens, next_token), axis=1)
    return tokens[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100, help="训练步数")
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps 必须大于或等于 0")

    mx.random.seed(42)
    train_text, validation_text = make_data()
    alphabet = sorted(set("".join(train_text)))
    char_to_id = {char: index for index, char in enumerate(alphabet)}

    def encode(examples):
        return mx.array([[char_to_id[char] for char in sentence] for sentence in examples])

    train, validation = encode(train_text), encode(validation_text)
    model = MiniGPT(len(alphabet), context_length=train.shape[1] - 1)
    optimizer = optim.AdamW(learning_rate=0.003)
    loss_and_grad = nn.value_and_grad(model, next_token_loss)

    parameter_count = sum(param.size for _, param in tree_flatten(model.parameters()))
    print(f"MLX 设备 {mx.default_device()}，参数量 {parameter_count:,}")
    print(f"训练 {len(train_text)} 句，验证 {len(validation_text)} 句，词表 {len(alphabet)} 个字符")

    def report(step):
        train_loss = next_token_loss(model, train).item()
        validation_loss = next_token_loss(model, validation).item()
        suffix_loss = next_token_loss(model, validation, suffix_length=4).item()
        print(f"第 {step:3d} 步  训练 loss {train_loss:.3f}  "
              f"验证 loss {validation_loss:.3f}  续写 loss {suffix_loss:.3f}")

    report(0)
    for step in range(1, args.steps + 1):
        _, grads = loss_and_grad(model, train)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        if step % 100 == 0 or step == args.steps:
            report(step)

    correct = 0
    for sentence, encoded in zip(validation_text, validation):
        prefix = encoded[:-4]
        result = complete(model, prefix, count=4)
        generated = "".join(alphabet[token] for token in result.tolist())
        correct += generated == sentence
        print(f"{sentence[:-4]} → {generated[len(prefix):]}（期望 {sentence[-4:]}）")
    print(f"未见组合的完整续写正确数 {correct}/{len(validation_text)}")


if __name__ == "__main__":
    main()
