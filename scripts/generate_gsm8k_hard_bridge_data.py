"""
Generate harder bridge data for GSM8K tool-use SFT.

The focus is not arithmetic execution, but problem-to-expression modeling:
ratios, fractions/percentages, time intervals, and multi-entity bookkeeping.
"""

import argparse
import json
import os
import random
from collections import Counter


def part(prefix, expr, result, after=""):
    return [
        {"type": "text", "text": prefix},
        {"type": "python", "text": expr},
        {"type": "python_output", "text": str(result)},
        {"type": "text", "text": after},
    ]


def make_row(prompt, steps, answer, kind, split):
    content = []
    for step in steps:
        content.extend(part(*step))
    content.append({"type": "text", "text": f"\nThe final answer is:\n\n#### {answer}"})
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": content},
        ],
        "meta": {
            "source": "gsm8k_hard_bridge",
            "kind": kind,
            "split": split,
            "answer": str(answer),
            "num_tool_calls": len(steps),
            "exprs": [s[1] for s in steps],
        },
    }


def ratio_age(rng, split):
    a = rng.randint(2, 9)
    b = rng.randint(a + 1, 14)
    unit = rng.randint(4, 12)
    years = rng.randint(3, 15)
    total = (a + b) * unit
    older = b * unit
    answer = older + years
    names = rng.choice([("Darrell", "Allen"), ("Mia", "Noah"), ("Lily", "Amy")])
    prompt = (
        f"{names[0]} and {names[1]}'s ages are in the ratio {a}:{b}. "
        f"If their total age now is {total}, calculate {names[1]}'s age {years} years from now. "
        "Give the final answer after ####."
    )
    steps = [
        ("First compute the total number of ratio parts: ", f"{a}+{b}", a + b, f" There are {a+b} parts.\n"),
        ("Compute the value of one part: ", f"{total}/{a+b}", unit, f" One part is {unit} years.\n"),
        (f"Compute {names[1]}'s current age and add {years} years: ", f"{b}*{unit}+{years}", answer, f" {names[1]} will be {answer}."),
    ]
    return make_row(prompt, steps, answer, "ratio_age", split)


def percent_remaining(rng, split):
    total = rng.randint(20, 120)
    percent = rng.choice([10, 20, 25, 40, 50, 60, 75])
    lost = total * percent // 100
    extra = rng.randint(1, 30)
    answer = total - lost + extra
    item = rng.choice(["balloons", "stickers", "oranges", "cards"])
    prompt = (
        f"Sally has {total} {item}. A gust of wind causes {percent}% of them to be lost. "
        f"Later she finds {extra} more {item}. How many {item} does she have now? "
        "Give the final answer after ####."
    )
    steps = [
        ("Compute how many were lost: ", f"{total}*{percent}/100", lost, f" She lost {lost}.\n"),
        ("Subtract the lost amount: ", f"{total}-{lost}", total - lost, f" She has {total-lost} left.\n"),
        ("Add the newly found amount: ", f"{total-lost}+{extra}", answer, f" She now has {answer}."),
    ]
    return make_row(prompt, steps, answer, "percent_remaining", split)


def fraction_total(rng, split):
    denom = rng.choice([2, 3, 4, 5])
    numer = rng.randint(1, denom - 1)
    total = rng.randint(12, 120)
    total += (-total * numer) % denom
    used = total * numer // denom
    add = rng.randint(1, 40)
    answer = total - used + add
    item = rng.choice(["cups", "pages", "masks", "tickets"])
    prompt = (
        f"A box has {total} {item}. {numer}/{denom} of the {item} are used, then {add} more {item} are added. "
        f"How many {item} are there now? Give the final answer after ####."
    )
    steps = [
        ("Compute the used amount: ", f"{total}*{numer}/{denom}", used, f" {used} are used.\n"),
        ("Subtract the used amount: ", f"{total}-{used}", total - used, f" {total-used} remain.\n"),
        ("Add the new amount: ", f"{total-used}+{add}", answer, f" There are {answer} now."),
    ]
    return make_row(prompt, steps, answer, "fraction_total", split)


def sleep_interval(rng, split):
    sleep_start = 22 * 60
    wake = 6 * 60
    total_sleep_window = 8 * 60
    walk_start = rng.randint(1 * 60, 4 * 60)
    walk_len = rng.randint(10, 45)
    bathroom = rng.randint(3, 15)
    answer = total_sleep_window - walk_len - bathroom
    prompt = (
        "Kim usually sleeps from 10 p.m. to 6 a.m. One night she sleepwalks "
        f"for {walk_len} minutes and also wakes up for {bathroom} minutes to go to the bathroom. "
        "How many minutes did she sleep in bed? Give the final answer after ####."
    )
    steps = [
        ("Compute the full sleep window in minutes: ", "8*60", total_sleep_window, f" The full window is {total_sleep_window} minutes.\n"),
        ("Subtract sleepwalking minutes: ", f"{total_sleep_window}-{walk_len}", total_sleep_window - walk_len, f" {total_sleep_window-walk_len} minutes remain.\n"),
        ("Subtract bathroom minutes: ", f"{total_sleep_window-walk_len}-{bathroom}", answer, f" She slept {answer} minutes in bed."),
    ]
    return make_row(prompt, steps, answer, "sleep_interval", split)


def multi_entity(rng, split):
    a = rng.randint(5, 30)
    more = rng.randint(2, 20)
    fewer = rng.randint(1, 15)
    b = a + more
    c = b - fewer
    answer = a + b + c
    item = rng.choice(["stamps", "friends", "books", "buttons"])
    prompt = (
        f"Max has {a} red {item}. He has {more} more blue {item} than red {item}, "
        f"and {fewer} fewer green {item} than blue {item}. How many {item} does Max have in all? "
        "Give the final answer after ####."
    )
    steps = [
        ("Compute the blue amount: ", f"{a}+{more}", b, f" Blue amount is {b}.\n"),
        ("Compute the green amount: ", f"{b}-{fewer}", c, f" Green amount is {c}.\n"),
        ("Add all three amounts: ", f"{a}+{b}+{c}", answer, f" Max has {answer} in all."),
    ]
    return make_row(prompt, steps, answer, "multi_entity", split)


def cost_two_items(rng, split):
    n1, p1 = rng.randint(2, 12), rng.randint(2, 9)
    n2, p2 = rng.randint(2, 12), rng.randint(2, 9)
    answer = n1 * p1 + n2 * p2
    prompt = (
        f"An eraser costs ${p1} and a pencil costs ${p2}. "
        f"How much do {n1} erasers and {n2} pencils cost? Give the final answer after ####."
    )
    steps = [
        ("Compute the eraser cost: ", f"{n1}*{p1}", n1 * p1, f" Erasers cost {n1*p1}.\n"),
        ("Compute the pencil cost: ", f"{n2}*{p2}", n2 * p2, f" Pencils cost {n2*p2}.\n"),
        ("Add both costs: ", f"{n1*p1}+{n2*p2}", answer, f" Total cost is {answer}."),
    ]
    return make_row(prompt, steps, answer, "cost_two_items", split)


GENERATORS = [ratio_age, percent_remaining, fraction_total, sleep_interval, multi_entity, cost_two_items]


def generate_rows(size, seed, split):
    rng = random.Random(seed)
    rows = [rng.choice(GENERATORS)(rng, split) for _ in range(size)]
    rng.shuffle(rows)
    return rows


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows):
    return {
        "num_rows": len(rows),
        "kinds": dict(sorted(Counter(r["meta"]["kind"] for r in rows).items())),
        "tool_calls": dict(sorted(Counter(r["meta"]["num_tool_calls"] for r in rows).items())),
        "examples": rows[:5],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/math_tool")
    parser.add_argument("--train-size", type=int, default=30000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=456)
    args = parser.parse_args()

    train = generate_rows(args.train_size, args.seed, "train")
    val = generate_rows(args.val_size, args.seed + 10_000_000, "val")
    train_path = os.path.join(args.output_dir, "gsm8k_hard_bridge_train.jsonl")
    val_path = os.path.join(args.output_dir, "gsm8k_hard_bridge_val.jsonl")
    summary_path = os.path.join(args.output_dir, "gsm8k_hard_bridge_summary.json")
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "train": summarize(train), "val": summarize(val)}, f, ensure_ascii=False, indent=2)
    print(json.dumps({"train_path": train_path, "train_rows": len(train), "val_path": val_path, "val_rows": len(val), "summary_path": summary_path}, indent=2))


if __name__ == "__main__":
    main()
