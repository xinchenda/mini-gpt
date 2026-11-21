"""Save fixed-prompt generations and descriptive repetition statistics."""
import argparse
import json
from pathlib import Path
import torch
from .inference import load_artifact

PROMPTS = (
    "The old man opened the door and",
    "Alice looked at the",
    "In the middle of the night",
    "She could not understand why",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, tokenizer = load_artifact(args.artifact)
    results = []
    for prompt in PROMPTS:
        ids = tokenizer.encode(prompt).ids
        for temperature in (0, 0.7):
            torch.manual_seed(42)
            output = model.generate(torch.tensor([ids]), 64, temperature=temperature)
            completion = output[0, len(ids):].tolist()
            grams = [tuple(completion[i:i + 4]) for i in range(len(completion) - 3)]
            results.append(dict(prompt=prompt, temperature=temperature, seed=42,
                                generated_tokens=len(completion),
                                repeated_4gram_fraction=1 - len(set(grams)) / len(grams),
                                text=tokenizer.decode(output[0].tolist())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved {len(results)} samples; repetition is descriptive, not a fluency score")


if __name__ == "__main__":
    main()
