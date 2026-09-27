#!/usr/bin/env python3
"""
Bias-vector interpolation for Semantle-trained LoReFT models.

Both intervention types store per-word bias as a simple embedding table
and are handled identically:
  - linear  (LoreftPerWordBiasIntervention): b_w = word_bias[w]
  - vae     (DistributionalWordIntervention): b_w = word_mu[w]  (mean, no sampling)

Given two words (must both exist in the checkpoint's items.json), this script:
1. Looks up per-word bias vectors b_w1, b_w2.
2. Interpolates between them with lerp (default) or slerp (--interp_method).
3. At each step runs greedy + temperature sampling at T in INTERP_TEMPERATURES
   (n_samples each).
4. Computes a per-pair LIPZ continuity summary (``trajectory_analysis``) from
   count-weighted decode-embedding centroids per step.
5. Saves JSON results + the embed_sim_vs_t.png plot (embedding similarity to
   each endpoint per variant), and on molecule tasks tfs_vs_t.png, the same
   curves under Morgan/Tanimoto similarity.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from boreft.bias_tables import get_bias_vector, stack_bias_vectors
from boreft.eval.semantle import (
    DEFAULT_MAX_NEW_TOKENS,
    _load_model_and_items,
    _get_intervention,
    generate_text,
)
from boreft.data_utils import add_load_latest_argument, resolve_assistant_suffix, prompt_tokenization_from_cfg, load_merged_run_config
from boreft.task_config import task_instruction, task_supports_fingerprints
from boreft.intervention_marker import (
    content_span_from_cfg,
    intervention_token_id_from_cfg,
    validate_intervention_position,
)
from boreft.text_similarity import (
    embedding_sim_per_text,
    encode_texts_normalized,
    rdkit_map_path_for_cfg,
)
from boreft.text_display import PLOT_LABEL_MAX, PLOT_TITLE_LABEL_MAX, plot_label
from boreft.eval.eval_suite import temp_key, trajectory_step_summary

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INTERP_TEMPERATURES: tuple[float, ...] = (1.0, 1.5)

# Cap in-plot decode labels so temperature panels do not become a SMILES cloud.
_MAX_POINT_LABELS_GREEDY = 12
_MAX_POINT_LABELS_SAMPLED = 8

# Backward-compatible aliases (notebooks / external scripts).
_get_bias_vector = get_bias_vector
_stack_bias_vectors = stack_bias_vectors


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def safe_filename(s: str) -> str:
    """Collapse a target string to a path-safe token.

    Necessary for anything but plain words: SMILES contain ``/``, ``\\``, ``(``
    and ``#``, all of which break paths (and W&B metric namespaces).
    """
    s = re.sub(r"\s+", "_", s.strip())
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)[:120]


def _index_words_by_id(items: list[dict]) -> Dict[str, int]:
    return {str(it.get("word", it["target"])): int(it["id"]) for it in items}


def _bias_type_label(reft_model) -> str:
    iv = _get_intervention(reft_model)
    if hasattr(iv, "reparameterize"):
        return "vae"
    return "linear"


def _clusters_from_semantle_dir(
    semantle_dir: str,
    words: list[str],
    top_k: int = 100,
) -> Dict[str, List[str]]:
    """Return {word: [csv_stem, ...]} for each word found in the top_k of each CSV."""
    csv_paths = sorted(
        os.path.join(semantle_dir, fn)
        for fn in os.listdir(semantle_dir)
        if fn.endswith(".csv")
    )
    if not csv_paths:
        raise FileNotFoundError(f"No .csv files found under {semantle_dir}")

    hits: Dict[str, List[str]] = {w: [] for w in words}
    for path in csv_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        rows: List[Tuple[str, float]] = []
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append((row["Word"].strip(), float(row["Similarity"])))
        top_words = {w for w, _ in sorted(rows, key=lambda x: -x[1])[:top_k]}
        for w in words:
            if w in top_words:
                hits[w].append(stem)
    return hits


# ─────────────────────────────────────────────────────────────────────────────
# Bias interpolation (lerp / slerp)
# ─────────────────────────────────────────────────────────────────────────────

VALID_INTERP_METHODS = frozenset({"lerp", "slerp"})


def normalize_interp_method(method: str) -> str:
    """Return a validated interpolation method name."""
    normalized = method.strip().lower()
    if normalized not in VALID_INTERP_METHODS:
        allowed = ", ".join(sorted(VALID_INTERP_METHODS))
        raise ValueError(
            f"interp_method must be one of {{{allowed}}}, got {method!r}"
        )
    return normalized


def lerp_torch(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    """Linear interpolation in bias space."""
    return (1.0 - t) * v0 + t * v1


def slerp_torch(
    v0: torch.Tensor, v1: torch.Tensor, t: float, eps: float = 1e-7
) -> torch.Tensor:
    """Spherical linear interpolation — direction via slerp, magnitude via lerp."""
    n0, n1 = torch.linalg.norm(v0), torch.linalg.norm(v1)
    if n0 < eps or n1 < eps:
        return (1.0 - t) * v0 + t * v1

    v0u, v1u = v0 / n0, v1 / n1
    omega = torch.acos(torch.clamp(torch.dot(v0u, v1u), -1.0, 1.0))

    if torch.abs(omega) < eps:
        out_unit = (1.0 - t) * v0u + t * v1u
        out_unit = out_unit / (torch.linalg.norm(out_unit) + eps)
    else:
        sin_w = torch.sin(omega)
        out_unit = (
            torch.sin((1.0 - t) * omega) / sin_w * v0u
            + torch.sin(t * omega) / sin_w * v1u
        )

    return out_unit * ((1.0 - t) * n0 + t * n1)


def interpolate_bias_torch(
    v0: torch.Tensor,
    v1: torch.Tensor,
    t: float,
    method: str = "lerp",
) -> torch.Tensor:
    """Interpolate between bias vectors using ``method`` (``lerp`` or ``slerp``)."""
    if normalize_interp_method(method) == "lerp":
        return lerp_torch(v0, v1, t)
    return slerp_torch(v0, v1, t)


def _interp_variants(n_samples: int) -> list[dict]:
    """Fixed interpolation sampling variants (greedy + INTERP_TEMPERATURES)."""
    variants = [{"key": "greedy", "label": "Greedy", "do_sample": False}]
    for temp in INTERP_TEMPERATURES:
        variants.append(
            {
                "key": f"temperature_{temp}",
                "label": f"T={temp}",
                "do_sample": True,
                "temperature": temp,
                "n_samples": n_samples,
            }
        )
    return variants


def _style_axes(ax) -> None:
    ax.grid(True, linestyle="-", alpha=0.35, linewidth=0.6)
    ax.set_axisbelow(True)


def _interp_subplot_grid(n_variants: int) -> tuple[int, int, tuple[float, float]]:
    """Return (nrows, ncols, figsize) sized to ``n_variants`` (no empty panels)."""
    if n_variants <= 1:
        return 1, 1, (6.0, 4.5)
    if n_variants == 2:
        return 1, 2, (10.0, 4.5)
    if n_variants == 3:
        return 1, 3, (14.0, 4.5)
    return 2, 2, (12.0, 9.0)


def _sim_legend_label(endpoint: str) -> str:
    return f"sim(gen, {plot_label(endpoint, PLOT_TITLE_LABEL_MAX)!r})"


def _dominant_label_indices(
    texts: Sequence[str],
    *,
    max_labels: int,
) -> list[int]:
    """Indices where the dominant decode changes, subsampled if too many."""
    idxs: list[int] = []
    prev = None
    for i, text in enumerate(texts):
        if not text or text == prev:
            continue
        idxs.append(i)
        prev = text
    if len(idxs) <= max_labels:
        return idxs
    pick = np.unique(
        np.round(np.linspace(0, len(idxs) - 1, max_labels)).astype(int)
    )
    return [idxs[int(j)] for j in pick]


# ─────────────────────────────────────────────────────────────────────────────
# Core function (callable from run_full_eval.py or standalone via main())
# ─────────────────────────────────────────────────────────────────────────────


def run_interpolation(
    output_dir: str,
    word1: str,
    word2: str,
    *,
    model_name: str = "meta-llama/Llama-3.2-1B",
    layer: int = 13,
    low_rank_dim: int = 64,
    t_steps: int = 101,
    interp_method: str = "lerp",
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    n_samples: int = 10,
    top_p: float = 1.0,
    cache_dir: str | None = None,
    semantle_dir: str | None = None,
    semantle_top_k: int = 100,
    save_dir: str | None = None,
    position: str | None = None,
    train_vocab: Optional[set] = None,
    reft_model=None,
    tokenizer=None,
    prompt: str | None = None,
    items: list | None = None,
    assistant_suffix: str | None = None,
    from_chat_template: bool | None = None,
    load_latest: bool = False,
) -> Tuple[dict, Dict[str, str]]:
    """Run bias-vector interpolation between word1 and word2.

    Args:
        output_dir:    Checkpoint directory (items.json + intervenable_model/).
        word1, word2:  Target words to interpolate between.
        save_dir:      Where to write JSON + PNGs. Defaults to
                       output_dir/interpolation_runs/<word1>__to__<word2>_layerN.
        n_samples:     Temperature samples per step for each T variant.
        interp_method: ``lerp`` (linear in bias space) or ``slerp`` (direction
                       slerp + magnitude lerp).
        train_vocab:   Training vocabulary for per-step n_unseen counts.
        from_chat_template: When set, controls ``add_special_tokens`` during prompt
            tokenization. When ``None``, read from checkpoint ``use_chat_template``.

    Returns:
        results:  Full interpolation results dict (also written as JSON).
        plots:    Mapping of plot name → saved PNG path.
    """
    if position is None:
        cfg_p = os.path.join(output_dir, "intervention_config.json")
        if os.path.isfile(cfg_p):
            with open(cfg_p, encoding="utf-8") as f:
                position = json.load(f).get("position", "l1")
        else:
            position = "l1"

    position = validate_intervention_position(position)
    interp_method = normalize_interp_method(interp_method)

    merged_cfg = load_merged_run_config(output_dir)
    embed_task = str(merged_cfg.get("task", "semantle"))
    rdkit_map_path = rdkit_map_path_for_cfg(merged_cfg)
    intervention_token_id = intervention_token_id_from_cfg(merged_cfg)
    if from_chat_template is None:
        from_chat_template = prompt_tokenization_from_cfg(merged_cfg)

    if reft_model is None or tokenizer is None or prompt is None or items is None:
        print(f"[interp] Loading checkpoint: {output_dir}")
        reft_model, tokenizer, _, prompt, items = _load_model_and_items(
            output_dir,
            model_name,
            layer,
            low_rank_dim,
            cache_dir=cache_dir,
            use_word_bias=None,
            variance=None,
            load_latest=load_latest,
        )
        assistant_suffix = resolve_assistant_suffix(
            tokenizer,
            merged_cfg,
            task_instruction(embed_task, use_chat_template=True),
        )
    else:
        print(f"[interp] Using pre-loaded checkpoint: {output_dir}")
        if assistant_suffix is None:
            assistant_suffix = resolve_assistant_suffix(
                tokenizer,
                merged_cfg,
                task_instruction(embed_task, use_chat_template=True),
            )

    content_span = content_span_from_cfg(tokenizer, merged_cfg, model_name)

    word_to_id = _index_words_by_id(items)
    w1, w2 = str(word1), str(word2)
    for wname, wval in [("word1", w1), ("word2", w2)]:
        if wval not in word_to_id:
            raise KeyError(
                f"{wname}='{wval}' not found. Examples: {list(word_to_id.keys())[:10]}"
            )

    id1, id2 = int(word_to_id[w1]), int(word_to_id[w2])
    bias_label = _bias_type_label(reft_model)
    print(
        f"[interp] Intervention: {bias_label}  |  word1={w1!r} (id {id1})  word2={w2!r} (id {id2})"
    )

    if semantle_dir is not None:
        hits = _clusters_from_semantle_dir(semantle_dir, [w1, w2], top_k=semantle_top_k)
        print(
            f"[interp] Cluster membership — {w1!r}: {hits[w1] or ['(none)']}, {w2!r}: {hits[w2] or ['(none)']}"
        )

    # ── Bias vectors ──────────────────────────────────────────────────────────
    b1 = get_bias_vector(reft_model, id1)
    b2 = get_bias_vector(reft_model, id2)
    b1_t = torch.tensor(b1, device=DEVICE)
    b2_t = torch.tensor(b2, device=DEVICE)
    cos_end = torch.nn.functional.cosine_similarity(b1_t, b2_t, dim=0).item()
    print(
        f"[interp] ||b1||={np.linalg.norm(b1):.4f}  ||b2||={np.linalg.norm(b2):.4f}  cos={cos_end:.4f}"
    )

    # ── Interpolation trajectory ──────────────────────────────────────────────
    print(f"[interp] interp_method={interp_method}")
    t_values = np.linspace(0.0, 1.0, t_steps, dtype=np.float64).tolist()
    bias_ts: List[np.ndarray] = [
        interpolate_bias_torch(b1_t, b2_t, float(t), method=interp_method)
        .detach()
        .float()
        .cpu()
        .numpy()
        for t in t_values
    ]

    # ── Generation: 1 greedy + n_samples temperature samples per step ──────────
    if save_dir is None:
        save_dir = os.path.join(
            output_dir,
            "interpolation_runs",
            f"{safe_filename(w1)}__to__{safe_filename(w2)}_layer{layer}",
        )
    os.makedirs(save_dir, exist_ok=True)

    train_vocab = set(train_vocab) if train_vocab is not None else None
    variants = _interp_variants(n_samples)

    results: dict = {
        "output_dir": output_dir,
        "bias_type": bias_label,
        "interp_method": interp_method,
        "layer": layer,
        "word1": {"target": w1, "id": id1},
        "word2": {"target": w2, "id": id2},
        "endpoints": {
            "b1_norm": float(np.linalg.norm(b1)),
            "b2_norm": float(np.linalg.norm(b2)),
            "cos_b1_b2": float(cos_end),
        },
        "sampling": {
            "variants": [
                {
                    "key": v["key"],
                    "label": v["label"],
                    "do_sample": v["do_sample"],
                    **(
                        {"temperature": v["temperature"], "n_samples": v["n_samples"]}
                        if v["do_sample"]
                        else {}
                    ),
                }
                for v in variants
            ]
        },
        "t_values": t_values,
        "points": [],
    }

    print(
        f"[interp] Generating variants: greedy + temperature "
        f"{', '.join(str(t) for t in INTERP_TEMPERATURES)} "
        f"(n_samples={n_samples} each)"
    )

    variant_outputs: dict[str, list] = {v["key"]: [] for v in variants}

    for i, (t, bt_np) in enumerate(zip(t_values, bias_ts)):
        bt_t = torch.tensor(bt_np, device=DEVICE)
        b_sim1 = torch.nn.functional.cosine_similarity(bt_t, b1_t, dim=0).item()
        b_sim2 = torch.nn.functional.cosine_similarity(bt_t, b2_t, dim=0).item()
        bn = float(torch.linalg.norm(bt_t).item())

        point: dict = {
            "t": float(t),
            "bias_norm": bn,
            "b_sim_to_word1": float(b_sim1),
            "b_sim_to_word2": float(b_sim2),
        }

        for variant in variants:
            key = variant["key"]
            if not variant["do_sample"]:
                text = generate_text(
                    reft_model,
                    tokenizer,
                    prompt,
                    bt_np,
                    max_new_tokens=max_new_tokens,
                    use_sample=False,
                    position=position,
                    assistant_suffix=assistant_suffix,
                    from_chat_template=from_chat_template,
                    intervention_token_id=intervention_token_id,
                    content_span=content_span,
                )
                point[key] = {"text": text}
                variant_outputs[key].append(text)
            else:
                samples = [
                    generate_text(
                        reft_model,
                        tokenizer,
                        prompt,
                        bt_np,
                        max_new_tokens=max_new_tokens,
                        use_sample=True,
                        temperature=variant["temperature"],
                        top_p=top_p,
                        position=position,
                        assistant_suffix=assistant_suffix,
                        from_chat_template=from_chat_template,
                        intervention_token_id=intervention_token_id,
                        content_span=content_span,
                    )
                    for _ in range(n_samples)
                ]
                uniq = set(samples)
                n_unique = len(uniq)
                n_unseen = (
                    sum(1 for s in uniq if s not in train_vocab)
                    if train_vocab is not None
                    else None
                )
                point[key] = {
                    "samples": samples,
                    "n_unique": n_unique,
                    "n_unseen": n_unseen,
                }
                variant_outputs[key].append(samples)

        greedy_text = point["greedy"]["text"]
        temp_unique = point[f"temperature_{INTERP_TEMPERATURES[1]}"]["n_unique"]
        print(
            f"  [{i:03d}] t={t:.3f}  ||b||={bn:.4f}  b_sim1={b_sim1:.3f}  "
            f"b_sim2={b_sim2:.3f}  greedy={greedy_text!r}  "
            f"T=1.5 n_unique={temp_unique}"
        )
        results["points"].append(point)

    # ── Embedding similarity per variant ─────────────────────────────────────
    has_embed = False
    try:
        embedding_sim_per_text([w1], [w1], task=embed_task)
        has_embed = True
    except Exception:
        pass

    variant_series: dict[str, dict] = {}
    trajectory_analysis: dict[str, dict] = {}
    # Structural similarity to each endpoint, tracked next to the embedding cosine
    # so a path can be read for "drifts through similar-looking chemistry" and
    # "drifts through shared substructures" independently.
    want_tfs = task_supports_fingerprints(embed_task)
    if has_embed:
        print("[interp] Computing embedding similarity + continuity per variant")
        emb_w1 = encode_texts_normalized([w1], task=embed_task)[0]
        emb_w2 = encode_texts_normalized([w2], task=embed_task)[0]
        for variant in variants:
            key = variant["key"]
            s1_mean: list[float] = []
            s1_std: list[float] = []
            s2_mean: list[float] = []
            s2_std: list[float] = []
            f1_mean: list[float] = []
            f1_std: list[float] = []
            f2_mean: list[float] = []
            f2_std: list[float] = []
            r1_mean: list[float] = []
            r1_std: list[float] = []
            r2_mean: list[float] = []
            r2_std: list[float] = []
            n_unique_per_t: list[int] = []
            dominant_texts: list[str] = []
            # Per-step semantic position: count-weighted centroid of the step's
            # decode embeddings (greedy -> the single decode's embedding).
            centroids: list[np.ndarray] = []

            for outputs, pt in zip(variant_outputs[key], results["points"]):
                if not variant["do_sample"]:
                    texts_at_t = [outputs]
                    weights = np.ones(1, dtype=np.float64)
                    n_unique_per_t.append(1)
                    dominant_texts.append(outputs)
                else:
                    counts = Counter(outputs)
                    texts_at_t = list(counts.keys())
                    weights = np.asarray(
                        [counts[t_] for t_ in texts_at_t], dtype=np.float64
                    )
                    n_unique_per_t.append(len(texts_at_t))
                    dominant_texts.append(counts.most_common(1)[0][0])

                emb_gen = encode_texts_normalized(texts_at_t, task=embed_task)  # [k, D], L2-normalized
                s1 = emb_gen @ emb_w1
                s2 = emb_gen @ emb_w2
                s1_mean.append(float(np.mean(s1)))
                s1_std.append(float(np.std(s1)) if len(texts_at_t) > 1 else 0.0)
                s2_mean.append(float(np.mean(s2)))
                s2_std.append(float(np.std(s2)) if len(texts_at_t) > 1 else 0.0)
                centroids.append(np.average(emb_gen, axis=0, weights=weights))

                pt[key]["embed_sim_to_word1"] = s1_mean[-1]
                pt[key]["embed_sim_to_word2"] = s2_mean[-1]
                pt[key]["dominant_text"] = dominant_texts[-1]

                if want_tfs:
                    from boreft.chem import rdkit_similarity, tanimoto_similarity

                    for endpoint, m_acc, sd_acc in (
                        (w1, f1_mean, f1_std),
                        (w2, f2_mean, f2_std),
                    ):
                        tfs = [tanimoto_similarity(endpoint, x) for x in texts_at_t]
                        m_acc.append(float(np.mean(tfs)))
                        sd_acc.append(float(np.std(tfs)) if len(tfs) > 1 else 0.0)
                    pt[key]["tfs_to_word1"] = f1_mean[-1]
                    pt[key]["tfs_to_word2"] = f2_mean[-1]
                    for endpoint, m_acc, sd_acc in (
                        (w1, r1_mean, r1_std),
                        (w2, r2_mean, r2_std),
                    ):
                        rdkit_sims = [
                            rdkit_similarity(
                                endpoint, x, map_path=rdkit_map_path
                            )
                            for x in texts_at_t
                        ]
                        m_acc.append(float(np.mean(rdkit_sims)))
                        sd_acc.append(
                            float(np.std(rdkit_sims))
                            if len(rdkit_sims) > 1
                            else 0.0
                        )
                    pt[key]["rdkit_sim_to_word1"] = r1_mean[-1]
                    pt[key]["rdkit_sim_to_word2"] = r2_mean[-1]

            variant_series[key] = {
                "label": variant["label"],
                "do_sample": variant["do_sample"],
                "temperature": variant.get("temperature"),
                "to_word1": {"mean": s1_mean, "std": s1_std},
                "to_word2": {"mean": s2_mean, "std": s2_std},
                "n_unique_per_t": n_unique_per_t,
                "dominant_text_per_t": dominant_texts,
            }
            if want_tfs:
                variant_series[key]["tfs_to_word1"] = {"mean": f1_mean, "std": f1_std}
                variant_series[key]["tfs_to_word2"] = {"mean": f2_mean, "std": f2_std}
                variant_series[key]["rdkit_sim_to_word1"] = {
                    "mean": r1_mean,
                    "std": r1_std,
                }
                variant_series[key]["rdkit_sim_to_word2"] = {
                    "mean": r2_mean,
                    "std": r2_std,
                }

            vkey = (
                "greedy"
                if not variant["do_sample"]
                else temp_key(float(variant["temperature"]))
            )
            trajectory_analysis[vkey] = trajectory_step_summary(centroids, bias_ts)

        results["embedding_similarity"] = {
            "note": "mean/std over unique generated texts per t step, per variant",
            "variants": variant_series,
        }
        # LIPZ per-pair continuity summary (aggregated across pairs by run_full_eval).
        results["trajectory_analysis"] = trajectory_analysis

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = os.path.join(save_dir, "interpolation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[interp] Saved: {json_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plots: Dict[str, str] = {}
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[interp] matplotlib not available — skipping plots.")
        return results, plots

    t_arr = np.asarray(t_values, dtype=np.float64)

    if has_embed:
        p3 = _plot_embed_sim_vs_t(
            plt,
            t_arr,
            variant_series,
            w1,
            w2,
            n_samples,
            os.path.join(save_dir, "embed_sim_vs_t.png"),
        )
        plots["embed_sim_vs_t"] = p3
        print(f"[interp] Saved: {p3}")

        if want_tfs:
            p4 = _plot_embed_sim_vs_t(
                plt,
                t_arr,
                variant_series,
                w1,
                w2,
                n_samples,
                os.path.join(save_dir, "tfs_vs_t.png"),
                key1="tfs_to_word1",
                key2="tfs_to_word2",
                ylabel="Tanimoto similarity (ECFP4)",
            )
            plots["tfs_vs_t"] = p4
            print(f"[interp] Saved: {p4}")
            p5 = _plot_embed_sim_vs_t(
                plt,
                t_arr,
                variant_series,
                w1,
                w2,
                n_samples,
                os.path.join(save_dir, "rdkit_sim_vs_t.png"),
                key1="rdkit_sim_to_word1",
                key2="rdkit_sim_to_word2",
                ylabel="RDKit descriptor similarity",
            )
            plots["rdkit_sim_vs_t"] = p5
            print(f"[interp] Saved: {p5}")

    return results, plots


def _plot_variant_embed_sim(
    ax,
    t_arr: np.ndarray,
    series: dict,
    w1: str,
    w2: str,
    n_samples: int,
    *,
    label_dominant_texts: bool,
    key1: str = "to_word1",
    key2: str = "to_word2",
) -> None:
    """Plot one variant's similarity-to-each-endpoint curves on ``ax``.

    ``key1``/``key2`` select which similarity series to read, so the same curves
    can be drawn for the embedding cosine or for Morgan/Tanimoto.
    """
    from matplotlib.collections import LineCollection

    s1_mean = np.asarray(series[key1]["mean"], dtype=np.float64)
    s1_std = np.asarray(series[key1]["std"], dtype=np.float64)
    s2_mean = np.asarray(series[key2]["mean"], dtype=np.float64)
    s2_std = np.asarray(series[key2]["std"], dtype=np.float64)
    n_unique_arr = np.asarray(series["n_unique_per_t"], dtype=np.float64)
    dominant_texts = series["dominant_text_per_t"]

    denom = max(float(n_samples - 1), 1.0)
    frac = np.clip((n_unique_arr - 1.0) / denom, 0.0, 1.0)
    widths = 1.0 + 4.0 * frac

    for mean, std, color, label in (
        (s1_mean, s1_std, "#2166AC", w1),
        (s2_mean, s2_std, "#B2182B", w2),
    ):
        pts = np.column_stack([t_arr, mean]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(
            segs,
            linewidths=(widths[:-1] + widths[1:]) / 2.0,
            colors=color,
            alpha=0.72,
        )
        ax.add_collection(lc)
        ax.fill_between(t_arr, mean - std, mean + std, color=color, alpha=0.12)
        ax.plot([], [], color=color, lw=2.5, alpha=0.72, label=_sim_legend_label(label))

    ax.set_xlim(t_arr.min(), t_arr.max())
    ax.set_ylim(0, 1.08)

    if label_dominant_texts:
        max_labels = (
            _MAX_POINT_LABELS_SAMPLED
            if series.get("do_sample")
            else _MAX_POINT_LABELS_GREEDY
        )
        texts = []
        y_offset = 0.035
        for i in _dominant_label_indices(dominant_texts, max_labels=max_labels):
            label_text = dominant_texts[i]
            t = t_arr[i]
            # Anchor above the endpoint curve the mode is closer to at this t.
            if s1_mean[i] >= s2_mean[i]:
                y = float(s1_mean[i]) + y_offset
            else:
                y = float(s2_mean[i]) + y_offset
            texts.append(
                ax.text(
                    float(t),
                    y,
                    plot_label(label_text, PLOT_LABEL_MAX),
                    fontsize=6.5,
                    ha="center",
                    va="bottom",
                    color="0.15",
                )
            )
        if texts:
            try:
                from adjustText import adjust_text

                adjust_text(
                    texts,
                    ax=ax,
                    arrowprops=dict(arrowstyle="-", color="0.55", lw=0.45),
                    expand=(1.08, 1.2),
                    force_text=(0.4, 0.6),
                )
            except ImportError:
                pass
            # Keep headroom for repelled labels (do not clip back to 1.08).
            _, ymax = ax.get_ylim()
            if ymax > 1.08:
                ax.set_ylim(0, ymax * 1.04)

    _style_axes(ax)


def _plot_embed_sim_vs_t(
    plt,
    t_arr: np.ndarray,
    variant_series: dict[str, dict],
    w1: str,
    w2: str,
    n_samples: int,
    save_path: str,
    *,
    key1: str = "to_word1",
    key2: str = "to_word2",
    ylabel: str = "Embedding similarity",
) -> str:
    """Similarity to each endpoint vs t, one subplot per sampling variant."""
    from boreft.bo.plotting import save_plot

    keys = list(variant_series.keys())
    n_variants = len(keys)
    nrows, ncols, figsize = _interp_subplot_grid(n_variants)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    axes_flat = axes.ravel()

    xlabel = r"Interpolation $t$  ($0=w_1,\ 1=w_2$)"
    for ax, key in zip(axes_flat, keys):
        series = variant_series[key]
        _plot_variant_embed_sim(
            ax,
            t_arr,
            series,
            w1,
            w2,
            n_samples,
            label_dominant_texts=True,
            key1=key1,
            key2=key2,
        )
        title = series["label"]
        if series["do_sample"]:
            title += f"  (N={n_samples})"
        ax.set_title(title, fontsize=11)

    for ax in axes_flat[n_variants:]:
        ax.set_visible(False)

    for row in range(nrows):
        for col in range(ncols):
            ax = axes[row, col]
            if not ax.get_visible():
                continue
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=11)
            if row == nrows - 1:
                ax.set_xlabel(xlabel, fontsize=11)

    w1_disp = plot_label(w1, PLOT_TITLE_LABEL_MAX)
    w2_disp = plot_label(w2, PLOT_TITLE_LABEL_MAX)
    fig.suptitle(
        f"Bias interpolation:  {w1_disp!r} → {w2_disp!r}   [{ylabel}]\n"
        r"(line width $\propto$ diversity for temperature variants)",
        fontsize=12,
        y=1.02,
    )
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=2,
        fontsize=8,
        frameon=True,
    )
    save_plot(fig, save_path, dpi=200)
    plt.close(fig)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="Interpolate bias vectors between two Semantle words."
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Checkpoint dir (items.json + intervenable_model/).",
    )
    add_load_latest_argument(p)
    p.add_argument("--model-name", dest="model_name", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--layer", type=int, default=13)
    p.add_argument("--low_rank_dim", type=int, default=64)
    p.add_argument("--word1", required=True)
    p.add_argument("--word2", required=True)
    p.add_argument(
        "--t_steps", type=int, default=101, help="Number of interpolation steps."
    )
    p.add_argument(
        "--interp_method",
        default="lerp",
        choices=sorted(VALID_INTERP_METHODS),
        help="Bias interpolation method: lerp (default) or slerp.",
    )
    p.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Temperature samples per t step for each T variant (greedy always 1).",
    )
    p.add_argument(
        "--top-p",
        dest="top_p",
        type=float,
        default=1.0,
        help="Nucleus top-p for the temperature variants.",
    )
    p.add_argument("--cache_dir", default=None)
    p.add_argument(
        "--semantle_dir", default=None, help="CSV dir to report cluster membership."
    )
    p.add_argument("--semantle_top_k", type=int, default=100)
    p.add_argument(
        "--position",
        default=None,
        help="Intervention position (e.g. l1, f1). Default: read intervention_config.json or l1.",
    )
    args = p.parse_args()
    if args.position is not None:
        validate_intervention_position(args.position)

    run_interpolation(
        output_dir=args.output_dir,
        word1=args.word1,
        word2=args.word2,
        model_name=args.model_name,
        layer=args.layer,
        low_rank_dim=args.low_rank_dim,
        t_steps=args.t_steps,
        interp_method=args.interp_method,
        max_new_tokens=args.max_new_tokens,
        n_samples=args.n_samples,
        top_p=args.top_p,
        cache_dir=args.cache_dir,
        semantle_dir=args.semantle_dir,
        semantle_top_k=args.semantle_top_k,
        position=args.position,
        load_latest=bool(args.load_latest),
    )


if __name__ == "__main__":
    main()
