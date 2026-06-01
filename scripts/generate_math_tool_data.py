"""
Generate calculator warmup data for math/tool-use SFT.

Example:
python -m scripts.generate_math_tool_data \
  --output-dir data/math_tool \
  --direct-train 10000 \
  --word-train 20000
"""

import argparse
import json
import operator
import os
import random
from collections import Counter


OPS = [
    ("+", operator.add),
    ("-", operator.sub),
    ("*", operator.mul),
]


def eval_expr(expr):
    value = eval(expr, {"__builtins__": {}}, {})
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def assistant_with_tool(prefix, expr, result, suffix=None):
    suffix = suffix or f"\nThe final answer is:\n\n#### {result}"
    return [
        {"type": "text", "text": prefix},
        {"type": "python", "text": expr},
        {"type": "python_output", "text": result},
        {"type": "text", "text": suffix},
    ]


def make_conversation(prompt, expr, meta, prefix=None):
    result = eval_expr(expr)
    prefix = prefix or "I should calculate this exactly: "
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_with_tool(prefix, expr, result)},
        ],
        "meta": {**meta, "expr": expr, "answer": result},
    }


def direct_arithmetic(rng, split):
    kind = rng.choice(["binary", "mixed", "parentheses"])
    if kind == "binary":
        op, _ = rng.choice(OPS)
        a = rng.randint(2, 999 if op != "*" else 99)
        b = rng.randint(2, 999 if op != "*" else 99)
        if op == "-" and b > a:
            a, b = b, a
        expr = f"{a}{op}{b}"
        prompt_templates = [
            "What is {expr}? Give the final answer after ####.",
            "Calculate {expr}. Put the final answer after ####.",
            "Use the calculator tool to compute {expr}, then answer after ####.",
            "I need the exact value of {expr}.",
        ]
    elif kind == "mixed":
        a = rng.randint(2, 80)
        b = rng.randint(2, 40)
        c = rng.randint(1, 200)
        op = rng.choice(["+", "-"])
        expr = f"{a}*{b}{op}{c}"
        prompt_templates = [
            "Calculate {expr}. Give the final answer after ####.",
            "What is the value of {expr}?",
            "Please compute {expr} exactly.",
        ]
    else:
        a = rng.randint(2, 60)
        b = rng.randint(1, 60)
        c = rng.randint(2, 30)
        expr = f"({a}+{b})*{c}"
        prompt_templates = [
            "What is {expr}? Give the final answer after ####.",
            "Evaluate {expr}.",
            "Use exact arithmetic for {expr}.",
        ]

    prompt = rng.choice(prompt_templates).format(expr=expr)
    return make_conversation(prompt, expr, {"source": "synthetic_direct", "kind": kind, "split": split})


def word_problem(rng, split):
    kind = rng.choice(["remaining", "purchase", "pages", "groups", "tickets", "distance"])

    if kind == "remaining":
        boxes = rng.randint(2, 12)
        per_box = rng.randint(6, 30)
        used = rng.randint(1, boxes * per_box - 1)
        item = rng.choice(["pencils", "stickers", "marbles", "cards"])
        expr = f"{boxes}*{per_box}-{used}"
        prompt = f"There are {boxes} boxes with {per_box} {item} in each box. If {used} {item} are used, how many {item} remain? Give the final answer after ####."
        prefix = "First compute the total and subtract the used amount: "
    elif kind == "purchase":
        n1 = rng.randint(2, 12)
        p1 = rng.randint(2, 25)
        n2 = rng.randint(1, 10)
        p2 = rng.randint(2, 25)
        item1 = rng.choice(["notebooks", "sandwiches", "tickets", "markers"])
        item2 = rng.choice(["pens", "drinks", "stickers", "folders"])
        expr = f"{n1}*{p1}+{n2}*{p2}"
        prompt = f"Lena buys {n1} {item1} for ${p1} each and {n2} {item2} for ${p2} each. How much does she spend? Give the final answer after ####."
        prefix = "The total cost is the first cost plus the second cost: "
    elif kind == "pages":
        days = rng.randint(3, 14)
        per_day = rng.randint(5, 40)
        already = rng.randint(10, 150)
        expr = f"{already}+{days}*{per_day}"
        prompt = f"A student has already read {already} pages. She reads {per_day} pages per day for {days} days. How many pages has she read in total? Give the final answer after ####."
        prefix = "Add the pages already read to the new pages read: "
    elif kind == "groups":
        groups = rng.randint(2, 12)
        per_group = rng.randint(3, 25)
        extra = rng.randint(1, 50)
        expr = f"{groups}*{per_group}+{extra}"
        prompt = f"There are {groups} groups with {per_group} students in each group, plus {extra} extra students. How many students are there? Give the final answer after ####."
        prefix = "Compute the grouped students and add the extra students: "
    elif kind == "tickets":
        adult = rng.randint(1, 8)
        child = rng.randint(1, 8)
        adult_price = rng.randint(8, 30)
        child_price = rng.randint(3, 20)
        expr = f"{adult}*{adult_price}+{child}*{child_price}"
        prompt = f"A family buys {adult} adult tickets at ${adult_price} each and {child} child tickets at ${child_price} each. What is the total cost? Give the final answer after ####."
        prefix = "Calculate adult ticket cost plus child ticket cost: "
    else:
        speed = rng.randint(20, 80)
        hours = rng.randint(2, 9)
        rest = rng.randint(5, 60)
        expr = f"{speed}*{hours}+{rest}"
        prompt = f"A driver travels {speed} miles per hour for {hours} hours, then drives {rest} more miles. How many miles does the driver travel? Give the final answer after ####."
        prefix = "Compute the distance from speed and time, then add the extra distance: "

    return make_conversation(prompt, expr, {"source": "synthetic_word", "kind": kind, "split": split}, prefix=prefix)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows):
    sources = Counter(row["meta"]["source"] for row in rows)
    kinds = Counter(row["meta"]["kind"] for row in rows)
    return {
        "num_rows": len(rows),
        "sources": dict(sorted(sources.items())),
        "kinds": dict(sorted(kinds.items())),
        "examples": rows[:5],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/math_tool")
    parser.add_argument("--direct-train", type=int, default=10000)
    parser.add_argument("--word-train", type=int, default=20000)
    parser.add_argument("--direct-val", type=int, default=500)
    parser.add_argument("--word-val", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train_rows = [direct_arithmetic(rng, "train") for _ in range(args.direct_train)]
    train_rows += [word_problem(rng, "train") for _ in range(args.word_train)]
    val_rows = [direct_arithmetic(rng, "val") for _ in range(args.direct_val)]
    val_rows += [word_problem(rng, "val") for _ in range(args.word_val)]
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    train_path = os.path.join(args.output_dir, "calculator_warmup_train.jsonl")
    val_path = os.path.join(args.output_dir, "calculator_warmup_val.jsonl")
    summary_path = os.path.join(args.output_dir, "calculator_warmup_summary.json")
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    summary = {
        "seed": args.seed,
        "train_path": train_path,
        "val_path": val_path,
        "train": summarize(train_rows),
        "val": summarize(val_rows),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "train_path": train_path,
        "train_rows": len(train_rows),
        "val_path": val_path,
        "val_rows": len(val_rows),
        "summary_path": summary_path,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
