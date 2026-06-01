"""
Quick smoke test for calculator tool-use behavior.

Example:
python -m scripts.tool_use_smoke \
  --source=sft \
  --model-tag=d12_pretrain \
  --step=1000 \
  --device-type=cuda
"""

import argparse
import json
import re
from contextlib import nullcontext

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init
from nanochat.engine import Engine


ANSWER_RE = re.compile(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)")


PROMPTS = [
    {
        "id": "direct_mul",
        "prompt": "What is 347 * 28? Give the final answer after ####.",
        "answer": "9716",
    },
    {
        "id": "direct_mixed",
        "prompt": "Calculate 18 * 7 - 23. Give the final answer after ####.",
        "answer": "103",
    },
    {
        "id": "word_one_step",
        "prompt": "A box has 18 pencils. There are 7 boxes. If 23 pencils are used, how many pencils remain? Give the final answer after ####.",
        "answer": "103",
    },
    {
        "id": "word_cost",
        "prompt": "Lena buys 4 notebooks for $3 each and 2 pens for $5 each. How much does she spend? Give the final answer after ####.",
        "answer": "22",
    },
    {
        "id": "gsm8k_style",
        "prompt": "Mimi picked up 2 dozen seashells. Kyle found twice as many shells as Mimi. Leigh grabbed one-third of Kyle's shells. How many seashells did Leigh have? Give the final answer after ####.",
        "answer": "16",
    },
    {
        "id": "non_math",
        "prompt": "Briefly explain what a tokenizer does in an LLM.",
        "answer": None,
    },
]


def extract_answer(text):
    match = ANSWER_RE.search(text)
    if match is None:
        return None
    return match.group(1).replace(",", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--source", type=str, default="sft", choices=["sft", "rl"])
    parser.add_argument("-g", "--model-tag", type=str, default=None)
    parser.add_argument("-s", "--step", type=int, default=None)
    parser.add_argument("--device-type", type=str, default="", choices=["", "cuda", "cpu", "mps"])
    parser.add_argument("-d", "--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("-m", "--max-new-tokens", type=int, default=256)
    parser.add_argument("-t", "--temperature", type=float, default=0.0)
    parser.add_argument("-k", "--top-k", type=int, default=1)
    parser.add_argument("--jsonl", type=str, default=None, help="Optional path to write per-prompt JSONL results.")
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)

    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    python_start = tokenizer.encode_special("<|python_start|>")
    output_start = tokenizer.encode_special("<|output_start|>")

    results = []
    for item in PROMPTS:
        prompt_tokens = [bos, user_start, *tokenizer.encode(item["prompt"]), user_end, assistant_start]
        with autocast_ctx:
            generated, masks = engine.generate_batch(
                prompt_tokens,
                num_samples=1,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        completion_tokens = generated[0][len(prompt_tokens):]
        completion = tokenizer.decode(completion_tokens)
        pred = extract_answer(completion)
        has_python = python_start in completion_tokens
        has_output = output_start in completion_tokens
        correct = None if item["answer"] is None else pred == item["answer"]
        result = {
            "id": item["id"],
            "prompt": item["prompt"],
            "expected": item["answer"],
            "predicted": pred,
            "correct": correct,
            "has_python_call": has_python,
            "has_tool_output": has_output,
            "completion": completion,
        }
        results.append(result)

        print("=" * 100)
        print(f"[{item['id']}] expected={item['answer']} predicted={pred} correct={correct}")
        print(f"tool_call={has_python} tool_output={has_output}")
        print(completion)

    math_results = [r for r in results if r["expected"] is not None]
    tool_call_rate = sum(r["has_python_call"] for r in math_results) / len(math_results)
    tool_output_rate = sum(r["has_tool_output"] for r in math_results) / len(math_results)
    answer_acc = sum(bool(r["correct"]) for r in math_results) / len(math_results)
    non_math_calls_tool = [r for r in results if r["expected"] is None and r["has_python_call"]]

    print("=" * 100)
    print("SUMMARY")
    print(f"math_tool_call_rate: {tool_call_rate:.2%}")
    print(f"math_tool_output_rate: {tool_output_rate:.2%}")
    print(f"math_answer_accuracy: {answer_acc:.2%}")
    print(f"non_math_tool_calls: {len(non_math_calls_tool)}")
    print(f"checkpoint_meta: {json.dumps(meta, ensure_ascii=False, default=str)}")

    if args.jsonl is not None:
        with open(args.jsonl, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
