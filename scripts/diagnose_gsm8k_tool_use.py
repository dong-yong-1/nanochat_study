"""Sample GSM8K generations and count calculator-tool usage."""

import argparse
import json

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_init
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K, extract_answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="sft")
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--device-type", default="cuda")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--jsonl", required=True)
    args = parser.parse_args()

    _, _, _, _, device = compute_init(args.device_type)
    model, tokenizer, _ = load_model(
        args.source,
        device,
        phase="eval",
        model_tag=args.model_tag,
        step=args.step,
    )
    engine = Engine(model, tokenizer)
    task = GSM8K(subset="main", split="test", stop=args.limit)

    python_start = tokenizer.encode_special("<|python_start|>")
    output_start = tokenizer.encode_special("<|output_start|>")
    rows = []

    with torch.amp.autocast(device_type=args.device_type, dtype=torch.bfloat16):
        for i in range(len(task)):
            conv = task[i]
            prompt = tokenizer.render_for_completion(conv)
            results, _ = engine.generate_batch(
                prompt,
                num_samples=1,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            completion_tokens = results[0][len(prompt) :]
            completion = tokenizer.decode(completion_tokens)
            ref = extract_answer(conv["messages"][-1]["content"][-1]["text"])
            pred = extract_answer(completion)
            row = {
                "idx": i,
                "question": conv["messages"][0]["content"],
                "ref": ref,
                "pred": pred,
                "correct": pred == ref,
                "python_calls": completion_tokens.count(python_start),
                "tool_outputs": completion_tokens.count(output_start),
                "completion": completion,
            }
            rows.append(row)
            print("=" * 80)
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in [
                            "idx",
                            "ref",
                            "pred",
                            "correct",
                            "python_calls",
                            "tool_outputs",
                        ]
                    },
                    ensure_ascii=False,
                )
            )
            print(row["question"])
            print(row["completion"][:1000])

    with open(args.jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    correct = sum(row["correct"] for row in rows)
    tool_examples = sum(row["python_calls"] > 0 for row in rows)
    avg_calls = sum(row["python_calls"] for row in rows) / max(1, len(rows))
    print("=" * 80)
    print(
        json.dumps(
            {
                "saved": args.jsonl,
                "correct": correct,
                "total": len(rows),
                "tool_examples": tool_examples,
                "avg_python_calls": avg_calls,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
