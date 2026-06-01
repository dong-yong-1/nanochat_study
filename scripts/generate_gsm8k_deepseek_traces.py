"""Generate GSM8K decomposition tool traces with the DeepSeek chat API.

The script rewrites official GSM8K train examples into nanochat's structured
assistant format:

    text -> python -> python_output -> text -> ... -> #### answer

DeepSeek proposes the step explanations and arithmetic expressions. This script
computes every python_output locally and rejects rows whose final answer does not
match the official GSM8K answer.
"""

import argparse
import ast
import json
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction

import requests
from datasets import Dataset, load_dataset
from dotenv import load_dotenv

from tasks.gsm8k import extract_answer


SYSTEM_PROMPT = """You rewrite GSM8K math solutions into high-quality tool-use traces.
Return only valid JSON. Do not use Markdown.

Rules:
- Decompose the problem into clear variables and units.
- Each step must explain what quantity is being computed and why.
- Each expression must be a numeric Python arithmetic expression only.
- Allowed expression syntax: numbers, +, -, *, /, parentheses.
- Do not use variables, assignments, functions, comparisons, or units inside expressions.
- Keep the number of steps between 2 and 6.
- The final_answer must exactly match the official answer.
"""


USER_TEMPLATE = """Question:
{question}

Official GSM8K solution:
{answer}

Official final answer:
{final_answer}

Rewrite this as JSON with this exact schema:
{{
  "steps": [
    {{
      "explanation": "short explanation naming the variable and unit",
      "expr": "numeric Python arithmetic expression"
    }}
  ],
  "final_answer": "{final_answer}"
}}
"""


ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def normalize_number(value):
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return str(value.numerator)
        as_float = float(value)
        if as_float.is_integer():
            return str(int(as_float))
        return f"{as_float:.10f}".rstrip("0").rstrip(".")
    text = str(value).replace(",", "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def safe_eval_expr(expr):
    tree = ast.parse(expr, mode="eval")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError(f"unsupported constant: {node.value!r}")
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ALLOWED_UNARYOPS):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ALLOWED_BINOPS):
            left = visit(node.left)
            right = visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if right == 0:
                raise ValueError("division by zero")
            return left / right
        raise ValueError(f"unsupported expression syntax: {ast.dump(node)}")

    return normalize_number(visit(tree))


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def call_deepseek(api_key, base_url, model, question, answer, final_answer, timeout, max_retries):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    question=question,
                    answer=answer,
                    final_answer=final_answer,
                ),
            },
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"retryable HTTP {response.status_code}: {response.text[:300]}")
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return extract_json(content)
        except Exception as exc:
            last_error = exc
            sleep_s = min(30, 2**attempt)
            time.sleep(sleep_s)
    raise RuntimeError(f"DeepSeek call failed after {max_retries} retries: {last_error}")


def build_row(row, trace, source_index):
    question = row["question"]
    answer = row["answer"]
    final_answer = extract_answer(answer)
    if final_answer is None:
        raise ValueError("official answer is missing #### number")

    if normalize_number(trace.get("final_answer", "")) != normalize_number(final_answer):
        raise ValueError(f"final answer mismatch: {trace.get('final_answer')} != {final_answer}")

    steps = trace.get("steps")
    if not isinstance(steps, list) or not (2 <= len(steps) <= 6):
        raise ValueError("steps must be a list with length 2..6")

    content = []
    exprs = []
    outputs = []
    for i, step in enumerate(steps, 1):
        explanation = str(step.get("explanation", "")).strip()
        expr = str(step.get("expr", "")).strip()
        if not explanation or not expr:
            raise ValueError("step missing explanation or expr")
        result = safe_eval_expr(expr)
        exprs.append(expr)
        outputs.append(result)
        prefix = explanation
        if not prefix.endswith((".", ":", ";")):
            prefix += "."
        prefix += " "
        content.extend(
            [
                {"type": "text", "text": prefix},
                {"type": "python", "text": expr},
                {"type": "python_output", "text": result},
                {"type": "text", "text": f" This gives {result}.\n"},
            ]
        )

    content.append({"type": "text", "text": f"The final answer is:\n\n#### {normalize_number(final_answer)}"})
    return {
        "messages": [
            {
                "role": "user",
                "content": question + " Give the final answer after ####.",
            },
            {"role": "assistant", "content": content},
        ],
        "meta": {
            "source": "gsm8k_deepseek_trace",
            "source_index": source_index,
            "answer": normalize_number(final_answer),
            "num_tool_calls": len(steps),
            "exprs": exprs,
            "outputs": outputs,
        },
    }


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/math_tool")
    parser.add_argument("--name-prefix", default="gsm8k_deepseek_trace")
    parser.add_argument("--train-size", type=int, default=800)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="optional cap for smoke generation")
    parser.add_argument("--arrow-path", default=None, help="read a cached GSM8K train Arrow file directly")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key and not args.dry_run:
        raise SystemExit(f"Missing API key. Export {args.api_key_env}=... first.")

    if args.arrow_path:
        ds = Dataset.from_file(args.arrow_path)
    else:
        ds = load_dataset("openai/gsm8k", "main", split="train")
    indices = list(range(len(ds)))
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    total = args.train_size + args.val_size
    if args.limit is not None:
        total = min(total, args.limit)
    selected = indices[:total]

    train_path = os.path.join(args.output_dir, f"{args.name_prefix}_train.jsonl")
    val_path = os.path.join(args.output_dir, f"{args.name_prefix}_val.jsonl")
    reject_path = os.path.join(args.output_dir, f"{args.name_prefix}_rejects.jsonl")
    summary_path = os.path.join(args.output_dir, f"{args.name_prefix}_summary.json")

    train_rows = load_existing(train_path) if args.resume else []
    val_rows = load_existing(val_path) if args.resume else []
    done = {row["meta"]["source_index"] for row in train_rows + val_rows}
    counts = Counter({"train": len(train_rows), "val": len(val_rows), "rejected": 0})

    jobs = []
    for ordinal, source_index in enumerate(selected, 1):
        split = "train" if ordinal <= args.train_size else "val"
        if source_index in done:
            continue
        jobs.append((ordinal, split, source_index))

    def process_job(job):
        _, split, source_index = job
        row = ds[source_index]
        final_answer = extract_answer(row["answer"])
        if args.dry_run:
            return {
                "kind": "dry_run",
                "split": split,
                "source_index": source_index,
                "question": row["question"],
                "answer": final_answer,
            }
        try:
            trace = call_deepseek(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                question=row["question"],
                answer=row["answer"],
                final_answer=final_answer,
                timeout=args.timeout,
                max_retries=args.max_retries,
            )
            out = build_row(row, trace, source_index)
            return {"kind": "ok", "split": split, "source_index": source_index, "row": out}
        except Exception as exc:
            return {
                "kind": "reject",
                "source_index": source_index,
                "split": split,
                "error": str(exc),
                "question": row["question"],
                "answer": row["answer"],
            }

    if args.dry_run:
        for job in jobs:
            print(json.dumps(process_job(job), ensure_ascii=False))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_job, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                if result["kind"] == "ok":
                    out = result["row"]
                    append_jsonl(train_path if result["split"] == "train" else val_path, out)
                    counts[result["split"]] += 1
                    print(
                        json.dumps(
                            {
                                "status": "ok",
                                "split": result["split"],
                                "source_index": result["source_index"],
                                "num_tool_calls": out["meta"]["num_tool_calls"],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                else:
                    counts["rejected"] += 1
                    append_jsonl(reject_path, result)
                    print(
                        json.dumps(
                            {
                                "status": "reject",
                                "split": result["split"],
                                "source_index": result["source_index"],
                                "error": result["error"],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

    if not args.dry_run:
        train_rows = load_existing(train_path)
        val_rows = load_existing(val_path)
        summary = {
            "seed": args.seed,
            "model": args.model,
            "base_url": args.base_url,
            "train_path": train_path,
            "val_path": val_path,
            "reject_path": reject_path,
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "rejected_this_run": counts["rejected"],
            "tool_calls_train": dict(sorted(Counter(r["meta"]["num_tool_calls"] for r in train_rows).items())),
            "tool_calls_val": dict(sorted(Counter(r["meta"]["num_tool_calls"] for r in val_rows).items())),
            "examples": train_rows[:3],
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
