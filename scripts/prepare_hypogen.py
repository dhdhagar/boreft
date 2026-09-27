#!/usr/bin/env python
"""Build a hypogen training corpus from an AutoDiscovery MCTS dump.

Writes two files that must stay in lockstep, because training validates that every
target has a definition:

``<out>``
    ``hypothesis`` — read by ``HypoGenItem.load_csv``. Row order matters:
    ``--train-top-k`` takes a prefix as the trained vocabulary and everything past
    the head becomes the held-out pool for RECON_TEST and GENZ. Rows are therefore
    shuffled with ``--seed`` rather than left in source order.

``<definitions>``
    ``{"target": <hypothesis>, "definition": <experiment_plan.objective>}`` per row.
    Category labels are omitted in v1.

Walks every object that has a ``hypothesis`` key (top-level MCTS nodes and nested
``tried_experiments`` / ``untried_experiments``). Deduplicates by whitespace-collapsed
case-insensitive hypothesis text, keeping the first occurrence.

    python scripts/prepare_hypogen.py

The warmstart split uses the initial ``experiment_generator`` proposals in
``node_1_0.json`` (kept in generator order):

    python scripts/prepare_hypogen.py --generator-messages \
        data/hypogen/raw/gpt4o-offline-0-evo-fresh-fish-500_20260429-123054/node_1_0.json \
        --no-shuffle
    # → data/hypogen/evo-fresh-fish-warmstart/train.csv
    # → data/hypogen/evo-fresh-fish-warmstart/definitions.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from typing import Any, Iterator

_RAW_RUN = os.path.join(
    "data",
    "hypogen",
    "raw",
    "gpt4o-offline-0-evo-fresh-fish-500_20260429-123054",
)
DEFAULT_NODES = os.path.join(_RAW_RUN, "mcts_nodes.json")
DEFAULT_GENERATOR_MESSAGES = os.path.join(_RAW_RUN, "node_1_0.json")
DEFAULT_OUT = os.path.join("data", "hypogen", "evo-fresh-fish", "train.csv")
DEFAULT_DEFINITIONS = os.path.join(
    "data", "hypogen", "evo-fresh-fish", "definitions.jsonl"
)
DEFAULT_WARMSTART_OUT = os.path.join(
    "data", "hypogen", "evo-fresh-fish-warmstart", "train.csv"
)
DEFAULT_WARMSTART_DEFINITIONS = os.path.join(
    "data", "hypogen", "evo-fresh-fish-warmstart", "definitions.jsonl"
)

HYPOTHESIS_COLUMN = "hypothesis"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--nodes",
        default=DEFAULT_NODES,
        help="MCTS nodes JSON (list of node objects).",
    )
    p.add_argument("--out", default=DEFAULT_OUT, help="Output CSV path.")
    p.add_argument(
        "--definitions",
        default=DEFAULT_DEFINITIONS,
        help="Output definitions JSONL path.",
    )
    p.add_argument(
        "--generator-messages",
        default=None,
        metavar="PATH",
        help=(
            "Chat-log JSON (list of messages). When set, take hypotheses from "
            "experiment_generator payloads instead of walking --nodes. "
            f"Typical: {DEFAULT_GENERATOR_MESSAGES}"
        ),
    )
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    p.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Keep first-seen / generator order (used for the warmstart split).",
    )
    p.add_argument(
        "--n",
        type=int,
        default=None,
        help="Optional cap after shuffle (debug / smoke tests).",
    )
    return p.parse_args(argv)


def hypothesis_identity_key(text: str) -> str:
    """Same collapse as text-task exact-match: whitespace + case-insensitive."""
    return " ".join(str(text).strip().lower().split())


def iter_hypothesis_objects(obj: Any) -> Iterator[dict[str, Any]]:
    """Yield every dict that carries a non-empty ``hypothesis`` string."""
    if isinstance(obj, dict):
        hypothesis = obj.get("hypothesis")
        if isinstance(hypothesis, str) and hypothesis.strip():
            yield obj
        for value in obj.values():
            yield from iter_hypothesis_objects(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_hypothesis_objects(item)


def experiment_objective(obj: dict[str, Any]) -> str:
    plan = obj.get("experiment_plan")
    if isinstance(plan, dict):
        objective = plan.get("objective")
        if isinstance(objective, str) and objective.strip():
            return objective.strip()
    raise ValueError("hypothesis object has no experiment_plan.objective")


def collect_rows_from_generator_messages(
    messages: Any,
) -> tuple[list[dict[str, str]], int]:
    """Hypotheses from ``experiment_generator`` chat messages (e.g. node_1_0.json)."""
    if not isinstance(messages, list):
        raise ValueError("generator messages JSON must be a list of chat messages")
    experiments: list[dict[str, Any]] = []
    n_generator = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("name") != "experiment_generator":
            continue
        n_generator += 1
        raw = message.get("content")
        if isinstance(raw, str):
            payload = json.loads(raw)
        elif isinstance(raw, dict):
            payload = raw
        else:
            raise ValueError("experiment_generator message has no JSON content")
        batch = payload.get("experiments")
        if not isinstance(batch, list):
            raise ValueError("experiment_generator payload has no 'experiments' list")
        experiments.extend(exp for exp in batch if isinstance(exp, dict))
    if n_generator == 0:
        raise ValueError("no experiment_generator message found")
    return collect_rows(experiments)


def collect_rows(nodes: Any) -> tuple[list[dict[str, str]], int]:
    """Return ``({hypothesis, definition}, n_collisions)`` in first-seen order."""
    rows: list[dict[str, str]] = []
    seen: dict[str, str] = {}
    n_collisions = 0
    for obj in iter_hypothesis_objects(nodes):
        hypothesis = str(obj["hypothesis"]).strip()
        definition = experiment_objective(obj)
        key = hypothesis_identity_key(hypothesis)
        previous = seen.get(key)
        if previous is None:
            seen[key] = definition
            rows.append({"hypothesis": hypothesis, "definition": definition})
            continue
        if previous != definition:
            n_collisions += 1
    return rows, n_collisions


def write_outputs(rows: list[dict[str, str]], *, out: str, definitions: str) -> None:
    if not rows:
        raise ValueError("no rows to write")
    for path in (out, definitions):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[HYPOTHESIS_COLUMN])
        writer.writeheader()
        for row in rows:
            writer.writerow({HYPOTHESIS_COLUMN: row["hypothesis"]})

    with open(definitions, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    {"target": row["hypothesis"], "definition": row["definition"]},
                    ensure_ascii=False,
                )
                + "\n"
            )

    lengths = [len(r["definition"]) for r in rows]
    print(
        f"[hypogen] wrote {len(rows)} hypotheses → {out} "
        f"(column: {HYPOTHESIS_COLUMN})"
    )
    print(
        f"[hypogen] wrote {len(rows)} definitions → {definitions} "
        f"(objective chars: min={min(lengths)} "
        f"median={sorted(lengths)[len(lengths) // 2]} max={max(lengths)})"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.generator_messages
    if source:
        if not os.path.isfile(source):
            print(f"ERROR: {source} not found", file=sys.stderr)
            return 1
        with open(source, encoding="utf-8") as f:
            payload = json.load(f)
        try:
            rows, n_collisions = collect_rows_from_generator_messages(payload)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.out == DEFAULT_OUT:
            args.out = DEFAULT_WARMSTART_OUT
        if args.definitions == DEFAULT_DEFINITIONS:
            args.definitions = DEFAULT_WARMSTART_DEFINITIONS
    else:
        if not os.path.isfile(args.nodes):
            print(f"ERROR: {args.nodes} not found", file=sys.stderr)
            return 1
        with open(args.nodes, encoding="utf-8") as f:
            nodes = json.load(f)
        rows, n_collisions = collect_rows(nodes)
    if n_collisions:
        print(
            f"[hypogen] {n_collisions} hypotheses had more than one distinct "
            f"objective; kept the first occurrence",
            flush=True,
        )
    if not rows:
        print("ERROR: no hypotheses found", file=sys.stderr)
        return 1
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(rows)
    if args.n is not None:
        rows = rows[: args.n]
    write_outputs(rows, out=args.out, definitions=args.definitions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
