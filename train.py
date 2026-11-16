"""在 CPU 上从随机权重训练一个字符级 GPT。"""

import argparse

import torch
from torch import nn
from torch.nn import functional as F


def make_data():
    animals = [("小猫", "喵喵叫"), ("小狗", "汪汪叫"),
               ("小牛", "哞哞叫"), ("小羊", "咩咩叫")]
    places = ["院里", "屋里", "河边", "山上"]
    train, validation = [], []
    for animal_index, (animal, sound) in enumerate(animals):
        for place_index, place in enumerate(places):
            examples = validation if animal_index == place_index else train
            examples.append(f"{animal}在{place}{sound}。")
    return train, validation


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, context_length):
        super().__init__()
        width = 32
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(context_length, width)
        self.attention_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_output = nn.Linear(width, width)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 4 * width),
            nn.GELU(), nn.Linear(4 * width, width),
        )
        self.readout = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, vocab_size))

    def forward(self, tokens):
        positions = torch.arange(tokens.size(1), device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        query, key, value = self.qkv(self.attention_norm(hidden)).chunk(3, dim=-1)
        # 因果遮罩让每个位置只能参考自己及之前的字符。
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        hidden = hidden + self.attention_output(attended)
        hidden = hidden + self.feed_forward(hidden)
        return self.readout(hidden)


def next_token_loss(model, sequences, suffix_length=None):
    inputs, targets = sequences[:, :-1], sequences[:, 1:]
    logits = model(inputs)

    if suffix_length is not None:
        logits = logits[:, -suffix_length:, :]
        targets = targets[:, -suffix_length:]

    return F.cross_entropy(logits.transpose(1, 2), targets)


@torch.no_grad()
def complete(model, prefix, count):
    tokens = prefix.unsqueeze(0)
    for _ in range(count):
        logits = model(tokens[:, -model.context_length:])
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        tokens = torch.cat((tokens, next_token), dim=1)
    return tokens[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300, help="训练步数")
    parser.add_argument("--resume", help="从指定的 checkpoint 恢复训练")
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps 必须大于或等于 0")

    torch.manual_seed(42)
    torch.set_num_threads(1)
    train_text, validation_text = make_data()
    alphabet = sorted(set("".join(train_text)))
    char_to_id = {char: index for index, char in enumerate(alphabet)}

    def encode(examples):
        return torch.tensor([[char_to_id[char] for char in sentence] for sentence in examples])

    train, validation = encode(train_text), encode(validation_text)
    model = MiniGPT(len(alphabet), context_length=train.size(1) - 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        start_step = checkpoint["step"]
        if args.steps < start_step:
            parser.error("--steps 不能小于存档步数")

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    print(f"训练 {len(train)} 句，验证 {len(validation)} 句，词表 {len(alphabet)} 个字符")
    print(f"模型参数量 {sum(parameter.numel() for parameter in model.parameters()):,}")
    print("输入示例", train_text[0][:-1])
    print("目标示例", train_text[0][1:])

    @torch.no_grad()
    def report(step):
        train_loss = next_token_loss(model, train).item()
        validation_loss = next_token_loss(model, validation).item()
        print(f"第 {step:3d} 步  训练 loss {train_loss:.3f}  验证 loss {validation_loss:.3f}")
        suffix_loss = next_token_loss(model, validation, suffix_length=4).item()
        print(f"            验证续写 loss {suffix_loss:.3f}")
    report(start_step)
    for step in range(start_step + 1, args.steps + 1):
        optimizer.zero_grad()
        loss = next_token_loss(model, train)
        loss.backward()
        optimizer.step()
        if step == 100:
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                "checkpoint_100.pt",
            )
        if step % 100 == 0 or step == args.steps:
            report(step)

    correct = 0
    for sentence, encoded in zip(validation_text, validation):
        prefix = encoded[:-4]
        if sentence.startswith("小牛"):
            with torch.no_grad():
                logits = model(prefix.unsqueeze(0))[0, -1]
                probs = logits.softmax(dim=-1)
                print(f"首字概率  哞 {probs[char_to_id['哞']].item():.1%}"
                      f"  咩 {probs[char_to_id['咩']].item():.1%}")
                place_logits = model(prefix[:3].unsqueeze(0))[0, -1]
                place_probs = place_logits.softmax(dim=-1)
                print(f"地点首字概率  河 {place_probs[char_to_id['河']].item():.4%}")
        result = complete(model, prefix, count=4)
        generated = "".join(alphabet[token] for token in result.tolist())
        correct += generated == sentence
        print(f"{sentence[:-4]} → {generated[len(prefix):]}（期望 {sentence[-4:]}）")
    print(f"未见组合的完整续写正确数 {correct}/{len(validation)}")


if __name__ == "__main__":
    main()
