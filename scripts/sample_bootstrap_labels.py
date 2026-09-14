"""Split the bootstrap TSVs into a small correction sample and a held-out eval set.

    python -m scripts.sample_bootstrap_labels --dir .data/crnn_bootstrap

`scripts.bootstrap_crnn_dataset` writes one row per crop across every split -
thousands of rows over the whole dataset. Correcting all of them is not the
efficient path: a CRNN over an 11-symbol digit alphabet, or a modest set of
frequently repeated Persian names, converges on a few hundred good examples,
and the value of each additional correction drops fast after that.

This script does two things instead:

  1. Pulls every crop from the `test_` split out into eval_text.tsv /
     eval_digits.tsv, untouched. It is small - one card's worth of fields
     times ~19 cards - and it was never seen while training the field
     detector either, so once corrected it becomes a trustworthy, held-out
     accuracy number. Correct all of it; there is no shortcut worth taking
     on your only real measurement.
  2. Draws a stratified random sample from the remaining `train_`/`val_`
     rows into sample_text.tsv / sample_digits.tsv, capped per field so a
     field the detector rarely misses does not crowd out the others.
     Correct THIS, train on it, and only pull more from the untouched
     remainder if evaluating against eval_*.tsv turns up a specific weak spot.

Re-running for more data (`--round 2`, `--round 3`, ...) never touches a file
that already exists: every previously generated sample_*/eval_* file in the
directory is read back in and its rows excluded from the new draw, so a
second round tops up with fresh crops instead of asking you to correct
something you already corrected, or silently overwriting corrected work.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path


def read_tsv(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        rel, _, text = line.partition("\t")
        rows.append((rel, text))
    return rows


def write_tsv(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for rel, text in rows:
            handle.write(f"{rel}\t{text}\n")


def field_key(rel_path: str) -> str:
    """'crops/text/train_foo__first_name.png' -> 'first_name'

    Matches the `<split>_<image>__<field>.png` naming bootstrap_crnn_dataset
    writes; the field is whatever follows the last '__'.
    """
    return Path(rel_path).stem.rsplit("__", 1)[-1]


def split_name(rel_path: str) -> str:
    """'crops/text/train_foo__first_name.png' -> 'train'"""
    stem_without_field = Path(rel_path).stem.split("__", 1)[0]
    return stem_without_field.split("_", 1)[0]


def stratified_sample(rows: list[tuple[str, str]], target_total: int, seed: int) -> list[tuple[str, str]]:
    """Cap each field at roughly `target_total / field_count`, shuffled.

    Capping per field, not just taking the first N rows, is what stops a
    field the detector locates on nearly every image (birth_date: 469/469)
    from filling the whole sample while a rarer one (father_name: 448/469)
    gets left out.
    """
    by_field: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        by_field[field_key(row[0])].append(row)

    fields = sorted(by_field)
    if not fields:
        return []
    per_field_cap = max(1, target_total // len(fields))

    rng = random.Random(seed)
    sample: list[tuple[str, str]] = []
    for field in fields:
        pool = list(by_field[field])
        rng.shuffle(pool)
        sample.extend(pool[:per_field_cap])
    return sample


def existing_output_paths(directory: Path, bucket: str) -> set[str]:
    """Every path already claimed by a previous sample_/eval_ file for this bucket.

    Read back, not tracked separately, so this stays correct even if a file
    was hand-edited (rows deleted as illegible) after generation - a deleted
    row is not "already used" and can be redrawn in a later round.
    """
    used: set[str] = set()
    for pattern in (f"sample_{bucket}*.tsv", f"eval_{bucket}*.tsv"):
        for path in directory.glob(pattern):
            for rel, _ in read_tsv(path):
                used.add(rel)
    return used


def next_round_suffix(directory: Path, bucket: str) -> str:
    """'' for the first run (sample_text.tsv), '_r2', '_r3', ... after that."""
    if not (directory / f"sample_{bucket}.tsv").exists():
        return ""
    round_number = 2
    while (directory / f"sample_{bucket}_r{round_number}.tsv").exists():
        round_number += 1
    return f"_r{round_number}"


def process(directory: Path, bucket: str, target: int, seed: int) -> None:
    labels_path = directory / f"labels_{bucket}.tsv"
    if not labels_path.exists():
        print(f"[skip] {labels_path} not found")
        return

    rows = read_tsv(labels_path)
    already_used = existing_output_paths(directory, bucket)
    suffix = next_round_suffix(directory, bucket)
    is_first_round = suffix == ""

    fresh_rows = [row for row in rows if row[0] not in already_used]
    eval_rows = [row for row in fresh_rows if split_name(row[0]) == "test"]
    trainable_rows = [row for row in fresh_rows if split_name(row[0]) != "test"]
    sample_rows = stratified_sample(trainable_rows, target, seed)

    eval_path = directory / f"eval_{bucket}{suffix}.tsv"
    sample_path = directory / f"sample_{bucket}{suffix}.tsv"

    # Only the first round produces an eval file - it exists to be a single,
    # stable held-out measurement. A second "held-out" file drawn from
    # whatever happened to remain would not mean the same thing.
    if is_first_round:
        write_tsv(eval_path, eval_rows)
    write_tsv(sample_path, sample_rows)

    counts: dict[str, int] = defaultdict(int)
    for row in sample_rows:
        counts[field_key(row[0])] += 1

    print(f"[{bucket}] {len(rows)} total rows, {len(already_used)} already claimed by earlier rounds")
    if is_first_round:
        print(f"[{bucket}] {eval_path.name:<22} {len(eval_rows):5d} rows  (full test split - correct all of it)")
    print(f"[{bucket}] {sample_path.name:<22} {len(sample_rows):5d} rows  (correct this)")
    for field, count in sorted(counts.items()):
        print(f"    {field:<14} {count}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dir", default=".data/crnn_bootstrap", help="Folder holding labels_text.tsv / labels_digits.tsv"
    )
    parser.add_argument("--digits-target", type=int, default=300, help="Rows to sample across all digit fields")
    parser.add_argument("--text-target", type=int, default=500, help="Rows to sample across all text fields")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    directory = Path(args.dir)
    process(directory, "digits", args.digits_target, args.seed)
    process(directory, "text", args.text_target, args.seed)

    print(
        "\nCorrect the new sample_*.tsv file(s) above by hand (crops are unchanged,\n"
        "still under crops/text/ and crops/digits/ inside the same folder), then\n"
        "either point train_crnn.py at the new file alone or concatenate it onto\n"
        "the one you already corrected before training."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
