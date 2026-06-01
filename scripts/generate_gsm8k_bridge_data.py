"""
Generate bridge data between calculator warmup and GSM8K.

These examples are deliberately simpler and more regular than raw GSM8K,
but they require multiple calculator calls and intermediate result chaining.

Example:
python -m scripts.generate_gsm8k_bridge_data \
  --output-dir data/math_tool \
  --train-size 10000 \
  --val-size 500
"""

import argparse
import json
import os
import random
from collections import Counter


def tool_parts(steps):
    parts = []
    for prefix, expr, result, after in steps:
        parts.append({"type": "text", "text": prefix})
        parts.append({"type": "python", "text": expr})
        parts.append({"type": "python_output", "text": str(result)})
        if after:
            parts.append({"type": "text", "text": after})
    return parts


def make_row(prompt, steps, answer, kind, split):
    parts = tool_parts(steps)
    parts.append({"type": "text", "text": f"\nThe final answer is:\n\n#### {answer}"})
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": parts},
        ],
        "meta": {
            "source": "gsm8k_bridge",
            "kind": kind,
            "split": split,
            "answer": str(answer),
            "num_tool_calls": len(steps),
            "exprs": [expr for _, expr, _, _ in steps],
        },
    }


def seashell_fraction(rng, split):
    dozens = rng.randint(1, 5)
    multiplier = rng.randint(2, 5)
    divisor = rng.choice([2, 3, 4])
    fraction_name = {2: "half", 3: "third", 4: "fourth"}[divisor]
    mimi = dozens * 12
    kyle = mimi * multiplier
    answer = kyle // divisor
    prompt = (
        f"Mimi picked up {dozens} dozen seashells. Kyle found {multiplier} times as many shells as Mimi. "
        f"Leigh grabbed one-{fraction_name} of Kyle's shells. "
        "How many seashells did Leigh have? Give the final answer after ####."
    )
    steps = [
        (f"First compute Mimi's shells: ", f"{dozens}*12", mimi, f" Mimi has {mimi} shells.\n"),
        (f"Kyle has {multiplier} times as many: ", f"{mimi}*{multiplier}", kyle, f" Kyle has {kyle} shells.\n"),
        (f"Leigh gets one-{fraction_name} of Kyle's shells: ", f"{kyle}/{divisor}", answer, f" Leigh has {answer} shells."),
    ]
    return make_row(prompt, steps, answer, "seashell_fraction", split)


def buy_then_share(rng, split):
    packs = rng.randint(2, 9)
    per_pack = rng.randint(6, 18)
    extra = rng.randint(2, 30)
    friends = rng.choice([2, 3, 4, 5, 6])
    total = packs * per_pack + extra
    total += (-total) % friends
    answer = total // friends
    item = rng.choice(["stickers", "marbles", "candies", "cards"])
    prompt = (
        f"Ava buys {packs} packs of {item} with {per_pack} {item} in each pack. "
        f"She also gets {extra + ((- (packs * per_pack + extra)) % friends)} extra {item}. "
        f"She shares all the {item} equally among {friends} friends. "
        f"How many {item} does each friend get? Give the final answer after ####."
    )
    adjusted_extra = extra + ((- (packs * per_pack + extra)) % friends)
    steps = [
        ("First compute the number from packs: ", f"{packs}*{per_pack}", packs * per_pack, f" The packs contain {packs * per_pack} {item}.\n"),
        ("Add the extra amount: ", f"{packs * per_pack}+{adjusted_extra}", total, f" There are {total} {item} total.\n"),
        ("Divide equally among the friends: ", f"{total}/{friends}", answer, f" Each friend gets {answer} {item}."),
    ]
    return make_row(prompt, steps, answer, "buy_then_share", split)


def work_earn_spend(rng, split):
    rate = rng.randint(8, 35)
    hours1 = rng.randint(2, 8)
    hours2 = rng.randint(1, 6)
    earned1 = rate * hours1
    earned2 = rate * hours2
    total = earned1 + earned2
    spent = rng.randint(5, max(6, total - 1))
    answer = total - spent
    prompt = (
        f"Noah earns ${rate} per hour. He works {hours1} hours on Monday and {hours2} hours on Tuesday. "
        f"Then he spends ${spent}. How much money does he have left? Give the final answer after ####."
    )
    steps = [
        ("Compute Monday's earnings: ", f"{rate}*{hours1}", earned1, f" Monday earnings are {earned1}.\n"),
        ("Compute Tuesday's earnings: ", f"{rate}*{hours2}", earned2, f" Tuesday earnings are {earned2}.\n"),
        ("Add both days and subtract the spending: ", f"{earned1}+{earned2}-{spent}", answer, f" Noah has {answer} left."),
    ]
    return make_row(prompt, steps, answer, "work_earn_spend", split)


def pages_remaining(rng, split):
    total_pages = rng.randint(120, 420)
    days = rng.randint(3, 10)
    per_day = rng.randint(8, 35)
    already = rng.randint(10, 80)
    read_more = days * per_day
    read_total = already + read_more
    if read_total >= total_pages:
        total_pages = read_total + rng.randint(20, 100)
    answer = total_pages - read_total
    prompt = (
        f"A book has {total_pages} pages. Sofia already read {already} pages. "
        f"She reads {per_day} pages per day for {days} days. "
        "How many pages are left? Give the final answer after ####."
    )
    steps = [
        ("Compute how many pages she reads over those days: ", f"{per_day}*{days}", read_more, f" She reads {read_more} more pages.\n"),
        ("Add the pages already read: ", f"{already}+{read_more}", read_total, f" She has read {read_total} pages total.\n"),
        ("Subtract from the full book: ", f"{total_pages}-{read_total}", answer, f" There are {answer} pages left."),
    ]
    return make_row(prompt, steps, answer, "pages_remaining", split)


def boxes_sell_buy(rng, split):
    boxes = rng.randint(3, 12)
    per_box = rng.randint(8, 30)
    sold = rng.randint(5, boxes * per_box - 1)
    bought = rng.randint(3, 60)
    total = boxes * per_box
    after_sold = total - sold
    answer = after_sold + bought
    item = rng.choice(["oranges", "pencils", "toy cars", "cookies"])
    prompt = (
        f"A store has {boxes} boxes of {item} with {per_box} {item} in each box. "
        f"It sells {sold} {item} and later buys {bought} more {item}. "
        f"How many {item} does the store have now? Give the final answer after ####."
    )
    steps = [
        ("Compute the starting amount: ", f"{boxes}*{per_box}", total, f" The store starts with {total} {item}.\n"),
        ("Subtract the sold amount: ", f"{total}-{sold}", after_sold, f" After selling, it has {after_sold} {item}.\n"),
        ("Add the newly bought amount: ", f"{after_sold}+{bought}", answer, f" The store now has {answer} {item}."),
    ]
    return make_row(prompt, steps, answer, "boxes_sell_buy", split)


def classroom_groups(rng, split):
    classes = rng.randint(2, 8)
    per_class = rng.randint(15, 35)
    absent = rng.randint(1, 20)
    buses = rng.choice([2, 3, 4, 5])
    total = classes * per_class
    going = total - absent
    going += (-going) % buses
    adjusted_absent = total - going
    answer = going // buses
    absent_phrase = "1 student is absent" if adjusted_absent == 1 else f"{adjusted_absent} students are absent"
    prompt = (
        f"There are {classes} classes with {per_class} students in each class. "
        f"{absent_phrase}. The remaining students are split evenly across {buses} buses. "
        "How many students ride each bus? Give the final answer after ####."
    )
    steps = [
        ("Compute the total number of students: ", f"{classes}*{per_class}", total, f" There are {total} students total.\n"),
        ("Subtract absent students: ", f"{total}-{adjusted_absent}", going, f" There are {going} students going.\n"),
        ("Divide them evenly across buses: ", f"{going}/{buses}", answer, f" Each bus has {answer} students."),
    ]
    return make_row(prompt, steps, answer, "classroom_groups", split)


GENERATORS = [
    seashell_fraction,
    buy_then_share,
    work_earn_spend,
    pages_remaining,
    boxes_sell_buy,
    classroom_groups,
]


def generate_rows(size, seed, split):
    rng = random.Random(seed)
    rows = []
    for _ in range(size):
        row = rng.choice(GENERATORS)(rng, split)
        rows.append(row)
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
        "kinds": dict(sorted(Counter(row["meta"]["kind"] for row in rows).items())),
        "tool_calls": dict(sorted(Counter(row["meta"]["num_tool_calls"] for row in rows).items())),
        "examples": rows[:5],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/math_tool")
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    train_rows = generate_rows(args.train_size, args.seed, "train")
    val_rows = generate_rows(args.val_size, args.seed + 10_000_000, "val")

    train_path = os.path.join(args.output_dir, "gsm8k_bridge_train.jsonl")
    val_path = os.path.join(args.output_dir, "gsm8k_bridge_val.jsonl")
    summary_path = os.path.join(args.output_dir, "gsm8k_bridge_summary.json")
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
