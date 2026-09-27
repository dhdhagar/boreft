#!/usr/bin/env python3
"""Serve the interactive BOReFT bias-subspace explorer."""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import uvicorn

from boreft.data_utils import add_load_latest_argument
from boreft.viz_server import VizRuntime, create_app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive 2D explorer and chat UI for a BOReFT checkpoint."
    )
    parser.add_argument("--checkpoint-dir", required=True)
    add_load_latest_argument(parser)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--low-rank-dim", type=int, default=None)
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--semantle-dir", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address (keep 0.0.0.0 when using an SSH tunnel).",
    )
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    runtime = VizRuntime.from_checkpoint(
        args.checkpoint_dir,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        layer=args.layer,
        low_rank_dim=args.low_rank_dim,
        torch_dtype=args.torch_dtype,
        semantle_dir=args.semantle_dir,
        top_k=args.top_k,
        load_latest=bool(args.load_latest),
    )
    app = create_app(runtime)
    print(
        f"[viz] Open http://127.0.0.1:{args.port} through your SSH tunnel.",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
