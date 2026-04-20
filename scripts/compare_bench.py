"""
Compare two inference benchmark JSON files produced by scripts.bench_infer.

Example:
python -m scripts.compare_bench \
    --before runs/bench_before_kvcache.json \
    --after runs/bench_after_kvcache.json
"""

import argparse
import json


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_results(rows):
    indexed = {}
    for row in rows:
        key = (row["prompt_len"], row["decode_len"], row["mode"])
        indexed[key] = row
    return indexed


def speedup(before_value, after_value):
    if before_value in (None, 0) or after_value is None:
        return None
    return before_value / after_value


def throughput_gain(before_value, after_value):
    if before_value in (None, 0) or after_value is None:
        return None
    return after_value / before_value


def print_stage_table(shared_keys, before_rows, after_rows):
    header = (
        f"{'prompt':>8} {'decode':>8} {'before_total':>13} {'after_total':>12} "
        f"{'total_x':>9} {'before_dec_t/s':>15} {'after_dec_t/s':>14} "
        f"{'decode_x':>9} {'before_mem':>12} {'after_mem':>11}"
    )
    print(header)
    print("-" * len(header))
    for key in shared_keys:
        before = before_rows[key]
        after = after_rows[key]
        total_x = speedup(before["total_ms_mean"], after["total_ms_mean"])
        decode_x = throughput_gain(before["decode_tok_s_mean"], after["decode_tok_s_mean"])
        print(
            f"{key[0]:8d} {key[1]:8d} "
            f"{before['total_ms_mean']:13.2f} {after['total_ms_mean']:12.2f} "
            f"{(total_x if total_x is not None else float('nan')):9.2f} "
            f"{(before['decode_tok_s_mean'] if before['decode_tok_s_mean'] is not None else float('nan')):15.2f} "
            f"{(after['decode_tok_s_mean'] if after['decode_tok_s_mean'] is not None else float('nan')):14.2f} "
            f"{(decode_x if decode_x is not None else float('nan')):9.2f} "
            f"{(before['peak_mem_delta_mib_mean'] if before['peak_mem_delta_mib_mean'] is not None else float('nan')):12.2f} "
            f"{(after['peak_mem_delta_mib_mean'] if after['peak_mem_delta_mib_mean'] is not None else float('nan')):11.2f}"
        )


def print_engine_table(shared_keys, before_rows, after_rows):
    header = (
        f"{'prompt':>8} {'decode':>8} {'before_ms':>12} {'after_ms':>12} "
        f"{'latency_x':>10} {'before_gen_t/s':>16} {'after_gen_t/s':>15} "
        f"{'throughput_x':>13} {'before_mem':>12} {'after_mem':>11}"
    )
    print(header)
    print("-" * len(header))
    for key in shared_keys:
        before = before_rows[key]
        after = after_rows[key]
        latency_x = speedup(before["engine_total_ms_mean"], after["engine_total_ms_mean"])
        throughput_x = throughput_gain(before["engine_gen_tok_s_mean"], after["engine_gen_tok_s_mean"])
        print(
            f"{key[0]:8d} {key[1]:8d} "
            f"{before['engine_total_ms_mean']:12.2f} {after['engine_total_ms_mean']:12.2f} "
            f"{(latency_x if latency_x is not None else float('nan')):10.2f} "
            f"{(before['engine_gen_tok_s_mean'] if before['engine_gen_tok_s_mean'] is not None else float('nan')):16.2f} "
            f"{(after['engine_gen_tok_s_mean'] if after['engine_gen_tok_s_mean'] is not None else float('nan')):15.2f} "
            f"{(throughput_x if throughput_x is not None else float('nan')):13.2f} "
            f"{(before['peak_mem_delta_mib_mean'] if before['peak_mem_delta_mib_mean'] is not None else float('nan')):12.2f} "
            f"{(after['peak_mem_delta_mib_mean'] if after['peak_mem_delta_mib_mean'] is not None else float('nan')):11.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Compare two inference benchmark JSON files")
    parser.add_argument("--before", type=str, required=True)
    parser.add_argument("--after", type=str, required=True)
    args = parser.parse_args()

    before_data = load_json(args.before)
    after_data = load_json(args.after)

    before_rows = index_results(before_data["results"])
    after_rows = index_results(after_data["results"])
    shared_keys = sorted(set(before_rows) & set(after_rows))
    if not shared_keys:
        raise ValueError("No overlapping benchmark rows found between the two files")

    before_mode = before_data["mode"]
    after_mode = after_data["mode"]
    if before_mode != after_mode:
        raise ValueError(f"Mode mismatch: before={before_mode}, after={after_mode}")

    print(f"Before: {args.before}")
    print(f"After : {args.after}")
    print(f"Mode  : {before_mode}")
    print()

    if before_mode == "stages":
        print_stage_table(shared_keys, before_rows, after_rows)
    else:
        print_engine_table(shared_keys, before_rows, after_rows)


if __name__ == "__main__":
    main()
