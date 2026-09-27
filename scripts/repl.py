#!/usr/bin/env python3
"""
Interactive REPL for a base HuggingFace model or a BOReFT checkpoint.

Loads a model once, then accepts prompts in a loop. In checkpoint mode you can
switch between base generation, zero-bias intervention, and per-word intervention
without reloading.

Usage:
  # Base instruct model (no checkpoint)
  python scripts/repl.py \\
    --model-name meta-llama/Llama-3.2-1B-Instruct \\
    --cache-dir ~/.cache/huggingface

  # Checkpoint — starts in base mode (no intervention hooks)
  python scripts/repl.py --checkpoint-dir outputs/your-run

  # Checkpoint — start with a trained target word
  python scripts/repl.py \\
    --checkpoint-dir outputs/your-run \\
    --target computer

  # Molopt — print SMILES validity after each decode
  python scripts/repl.py \\
    --checkpoint-dir outputs/your-run \\
    --check-smiles

REPL commands:
  /help            Show this help text
  /quit, /exit     Exit
  /clear           Clear chat history
  /chat            Accumulate multi-turn context
  /no-chat         Single-turn (no history accumulation)
  /target <word>   Intervention for a checkpoint vocabulary word
  /target-list     List checkpoint vocabulary words (/targets)
  /target-test-list
                   List eval test-set words (bias-network checkpoints)
  /target-test <word>
                   Predict μ for a test-set word (ST embed or llm_encoder definition)
  /target-text <text>
                   Predict μ for arbitrary text (ST embed, or llm_encoder with
                   last_instruction / instruction_mean over all encoded tokens)
  /learn <target> :: <definition>
                   Learn a bias vector for an unseen {target, definition} by
                   optimizing it directly (frozen network/R/W/LM) with the
                   checkpoint's training objective (CE + VAE KL + SDPO) plus an
                   anchor to the predicted μ. Persists to learned_biases.jsonl.
                   Optional trailing tokens: max-steps=N lr=F anchor=F.
  /learned-list    List persisted learned {target, definition} vectors
  /learned <target>
                   Load a persisted learned bias vector for intervention
  /temp <float>    Set sampling temperature (e.g. /temp 0.8)
  /temp off        Greedy decoding (no sampling)
  /system <text>   Chat-template system message (prepended each turn)
  /system <<       Multiline system prompt; end with a line containing only .
  /system off      Clear the system message
  /chat-template on|off
                   Force chat wrapping on or off (base mode only)
  /zero            Intervention hooks on, zero bias vector (no target word)
  /base            Base model only (no intervention hooks)
  /interp-method [lerp|slerp]
                   Show or set bias interpolation method (default: lerp)
  /interpolate [lerp|slerp] <w1> <w2> <n_steps> [<n_samples>]
                   Interpolate bias vectors between two checkpoint targets

/chat and /no-chat control history only. Chat-template rendering follows the
loaded checkpoint, or the tokenizer when no checkpoint is loaded. Qwen
base/CPT checkpoints often ship a chat_template even though the weights
were not instruction-tuned; use --no-chat-template (or /chat-template off)
and greedy decoding for completion-style SMILES probes.
Intervention modes inject the marker on the current user turn only.
--use-checkpoint-prompt ignores typed input and always uses the saved eval prompt.
--check-smiles reports whether the first token of each decode is valid SMILES,
RDKit chemistry issues when sanitization is what failed, and a SmiSelf
repair when that package is installed.
/system prepends a chat-template system turn; it is ignored without a chat
template and with --use-checkpoint-prompt. /system << reads additional lines
until a line containing only ``.``.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

try:
    import readline  # noqa: F401 — enables up/down arrow input history on Unix
except ImportError:
    pass

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from boreft.bias_tables import (
    bias_predict_kwargs,
    encoder_definitions_path,
    get_bias_vector,
    predict_bias_vectors_for_words,
    predict_bias_vectors_from_raw_texts,
)
from boreft.chem import (
    first_smiles_token,
    is_valid_smiles,
    repair_smiles,
    smiles_chemistry_problems,
)
from boreft.data_utils import (
    add_load_latest_argument,
    infer_torch_dtype_name,
    load_merged_run_config,
    resolve_torch_dtype,
    system_prompt_from_cfg,
)
from boreft.eval.eval_suite import (
    DEFAULT_BBOX_PCA_VAR,
    DEFAULT_TEST_N_SAMPLES,
    build_test_sets,
    normalize_text,
    task_csv_paths,
)
from boreft.eval.interpolate import (
    VALID_INTERP_METHODS,
    _index_words_by_id,
    interpolate_bias_torch,
    normalize_interp_method,
)
from boreft.learn_bias import (
    LearnBiasConfig,
    find_learned_bias,
    get_train_embeddings,
    learn_bias_for_pair,
    load_learned_biases,
)
from boreft.eval.semantle import (
    DEFAULT_MAX_NEW_TOKENS,
    generate_text,
    generate_texts_batch,
    load_eval_checkpoint,
)
from boreft.interactive import (
    GenerationMode,
    build_chat_prompt as _build_chat_prompt,
    build_checkpoint_prompt as _build_checkpoint_prompt,
    generate_base as _generate_base,
    prepend_system_message,
    strip_thinking_text,
    target_intervention_flags as _target_intervention_flags,
)
from boreft.text_similarity import definition_lookup_for_cfg

def _looks_like_local_path(name: str) -> bool:
    return (
        os.path.isabs(name)
        or name.startswith((".", "~"))
        or os.sep in name
        or (os.altsep is not None and os.altsep in name)
    )


def _resolve_model_name(model_name: str) -> str:
    """Use an existing local directory; otherwise keep a Hub id.

    Relative paths are tried from cwd and the repo root. A slash-separated
    string that is not a directory is not a Hub id (``mist/models/...`` has
    too many segments) — fail with the paths that were tried.
    """
    text = os.path.expanduser((model_name or "").strip())
    if not text:
        raise ValueError("model name is empty")
    candidates: list[str] = []
    if os.path.isabs(text):
        candidates.append(text)
    else:
        candidates.append(os.path.abspath(text))
        repo_path = os.path.abspath(os.path.join(REPO_ROOT, text))
        if repo_path not in candidates:
            candidates.append(repo_path)
    for path in candidates:
        if os.path.isdir(path):
            return path
    if _looks_like_local_path(text):
        tried = "\n  ".join(candidates)
        raise FileNotFoundError(
            f"local model directory not found: {model_name!r}\n"
            f"tried:\n  {tried}\n"
            "pass the absolute path to the folder that contains config.json"
        )
    return text


def _load_base_model(
    model_name: str,
    cache_dir: str | None,
    torch_dtype: str | None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_name = infer_torch_dtype_name(override=torch_dtype)
    load_dtype = resolve_torch_dtype(dtype_name)
    model_name = _resolve_model_name(model_name)
    local = os.path.isdir(model_name)

    tok_kwargs: dict = {"use_fast": True}
    model_kwargs: dict = {"dtype": load_dtype}
    if cache_dir and not local:
        tok_kwargs["cache_dir"] = cache_dir
        model_kwargs["cache_dir"] = cache_dir

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device == "cuda" else None,
        **model_kwargs,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return model, tokenizer, device, dtype_name, model_name


def _mode_label(
    *,
    has_checkpoint: bool,
    generation_mode: GenerationMode,
    target: str | None,
    intervention_label: str | None = None,
) -> str:
    if not has_checkpoint:
        return "base"
    if generation_mode == "base":
        return "base (no intervention hooks)"
    if generation_mode == "zero_bias":
        return "intervention (zero bias, no target word)"
    if intervention_label:
        return f"intervention ({intervention_label})"
    return f"intervention (target={target!r})"


def _print_target_list(word_to_id: dict[str, int]) -> None:
    words = sorted(word_to_id)
    print(f"[repl] {len(words)} targets:")
    for word in words:
        print(f"  {word}  (id={word_to_id[word]})")


def _print_target_test_list(
    test_words: list[str],
    *,
    interp_words: set[str],
) -> None:
    print(f"[repl] {len(test_words)} test-set words:")
    for word in test_words:
        tag = "interp" if word in interp_words else "extrap"
        print(f"  {word}  ({tag})")


def _sampling_label(*, do_sample: bool, temperature: float, top_p: float) -> str:
    if not do_sample:
        return "greedy"
    return f"sampled (temperature={temperature}, top_p={top_p})"


def _run_repl_interpolation(
    ckpt,
    *,
    word1: str,
    word2: str,
    n_steps: int,
    n_samples: int,
    word_to_id: dict[str, int],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    interp_method: str = "lerp",
) -> None:
    """Interpolate between two checkpoint targets and decode at each step."""
    interp_method = normalize_interp_method(interp_method)
    for word in (word1, word2):
        if word not in word_to_id:
            print(f"[repl] unknown target {word!r}")
            return

    id1, id2 = word_to_id[word1], word_to_id[word2]
    b1 = get_bias_vector(ckpt.reft_model, id1)
    b2 = get_bias_vector(ckpt.reft_model, id2)
    b1_t = torch.tensor(b1, dtype=torch.float32)
    b2_t = torch.tensor(b2, dtype=torch.float32)
    cos_end = torch.nn.functional.cosine_similarity(b1_t, b2_t, dim=0).item()

    cfg = ckpt.saved_cfg or {}
    position = cfg.get("position", "l1")
    t_values = (
        [i / (n_steps - 1) for i in range(n_steps)] if n_steps > 1 else [0.0]
    )

    use_sample = do_sample or n_samples > 1
    print(
        f"[interp] {word1!r} -> {word2!r}  method={interp_method}  steps={n_steps}  "
        f"n_samples={n_samples}  cos(b1,b2)={cos_end:.4f}  "
        f"sampling={_sampling_label(do_sample=use_sample, temperature=temperature, top_p=top_p)}"
    )

    gen_common = dict(
        max_new_tokens=max_new_tokens,
        use_sample=use_sample,
        temperature=temperature,
        top_p=top_p,
        position=position,
        assistant_suffix=ckpt.assistant_suffix,
        from_chat_template=ckpt.from_chat_template,
        intervention_token_id=ckpt.intervention_token_id,
        content_span=ckpt.content_span,
    )

    for t in t_values:
        bt_np = (
            interpolate_bias_torch(b1_t, b2_t, float(t), method=interp_method)
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        if n_samples == 1:
            text = generate_text(
                ckpt.reft_model,
                ckpt.tokenizer,
                ckpt.prompt,
                bt_np,
                **gen_common,
            )
            print(f"  t={t:.3f}  {text}")
        else:
            samples = generate_texts_batch(
                ckpt.reft_model,
                ckpt.tokenizer,
                ckpt.prompt,
                bt_np,
                n_samples,
                **gen_common,
            )
            n_unique = len(set(samples))
            joined = " | ".join(samples)
            print(f"  t={t:.3f}  n_unique={n_unique}/{n_samples}  {joined}")
    print()


def _parse_learn_options(opts: str, config: LearnBiasConfig) -> LearnBiasConfig:
    """Apply ``key=value`` overrides (max-steps, lr, anchor, sim-tol) to a config."""
    for tok in opts.split():
        if "=" not in tok:
            raise ValueError(f"bad option {tok!r} (expected key=value)")
        key, _, val = tok.partition("=")
        key = key.strip().lower().replace("_", "-")
        if key == "max-steps":
            config.max_steps = int(val)
        elif key == "lr":
            config.lr = float(val)
        elif key == "anchor":
            config.anchor_weight = float(val)
        elif key == "sim-tol":
            config.sim_tol = float(val)
        else:
            raise ValueError(f"unknown option {key!r}")
    return config


def _run_repl_learn(
    ckpt,
    *,
    checkpoint_dir: str,
    target: str,
    definition: str,
    options: str,
) -> np.ndarray | None:
    """Learn and persist a bias vector for one {target, definition}; return μ."""
    try:
        config = _parse_learn_options(options, LearnBiasConfig())
    except ValueError as exc:
        print(f"[repl] {exc}")
        return None

    print(
        f"[learn] target={target!r} definition={definition!r} "
        f"max_steps={config.max_steps} lr={config.lr} anchor={config.anchor_weight}"
    )
    try:
        result = learn_bias_for_pair(
            ckpt,
            target=target,
            definition=definition,
            checkpoint_dir=checkpoint_dir,
            config=config,
            show_progress=True,
            persist=True,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"[repl] learn failed: {exc}")
        return None

    status = "converged" if result.converged else "stopped (max steps)"
    print(
        f"[repl] {status} in {result.steps} steps  "
        f"loss={result.final_loss:.4f}  sim={result.final_sim:.3f}  "
        f"decode={result.decode!r}"
    )
    print(
        f"[repl] mu_pred (pre-training): decode={result.pred_decode!r} "
        f"sim={result.pred_sim:.3f}  ->  learned: decode={result.decode!r} "
        f"sim={result.final_sim:.3f}"
    )
    _print_learned_placement(result)
    if result.persist_path:
        print(f"[repl] saved learned bias -> {result.persist_path}")
    return result.mu


def _print_learned_placement(result) -> None:
    """Print interp/extrap and nearest-train-target placement of a learned bias."""
    if result.classification is None and result.closest_train_word is None:
        return
    parts: list[str] = []
    if result.classification is not None:
        parts.append(f"{result.classification} w.r.t. train set")
    if result.closest_train_word is not None:
        neighbor = f"closest train={result.closest_train_word!r}"
        if result.bias_distance is not None:
            neighbor += f" (bias L2={result.bias_distance:.3f}"
            if result.embed_sim_to_closest is not None:
                neighbor += f", embed sim={result.embed_sim_to_closest:.3f}"
            neighbor += ")"
        parts.append(neighbor)
    print(f"[repl] placement: {'  '.join(parts)}")


def _print_system_prompt(system_prompt: str | None) -> None:
    if not system_prompt:
        print("[repl] system=off")
        return
    if "\n" in system_prompt:
        print("[repl] system:")
        print(system_prompt)
        return
    print(f"[repl] system={system_prompt!r}")


def _read_multiline_block(*, end_tokens: tuple[str, ...] = (".", "/end")) -> str | None:
    """Read lines until a terminator. ``None`` means the user cancelled."""
    print("[repl] enter system prompt; end with a line containing only '.'")
    lines: list[str] = []
    while True:
        try:
            line = input("... ")
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if line.strip() in end_tokens:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _build_base_prompt(
    tokenizer,
    user_text: str,
    history: list[dict[str, str]],
    *,
    use_chat_template: bool,
    accumulate_history: bool,
    system_prompt: str | None = None,
) -> tuple[str, bool]:
    if use_chat_template:
        messages = (
            [*history, {"role": "user", "content": user_text}]
            if accumulate_history
            else [{"role": "user", "content": user_text}]
        )
        messages = prepend_system_message(messages, system_prompt)
        return _build_chat_prompt(tokenizer, messages), True
    return user_text, False


def _smiles_status_line(text: str) -> str:
    """Parenthetical molecule status for one decode (validity now; oracles later)."""
    candidate = first_smiles_token(strip_thinking_text(text))
    valid = bool(candidate) and is_valid_smiles(candidate)
    fields = [f"valid: {'true' if valid else 'false'}"]
    if not valid:
        issues = smiles_chemistry_problems(candidate)
        if issues:
            fields.append(f"issues: {'; '.join(issues)}")
        repaired = repair_smiles(candidate)
        if repaired:
            fields.append(f"smiself: {repaired}")
    return f"({', '.join(fields)})"


def _print_help(*, has_checkpoint: bool, ckpt) -> None:
    """Print module docstring and, for checkpoints, the saved training/eval prompt."""
    print(__doc__.strip())
    if has_checkpoint and ckpt is not None:
        print()
        print(
            "[repl] Canonical training/eval prompt "
            "(used with --use-checkpoint-prompt and /interpolate):"
        )
        print("-" * 72)
        print(ckpt.prompt)
        print("-" * 72)


def main() -> None:
    p = argparse.ArgumentParser(description="Interactive model REPL.")
    p.add_argument(
        "--model-name",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model ID (base mode, or fallback for checkpoint).",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="HF cache dir (empty string = HF default).",
    )
    p.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Training output dir with intervenable_model/ (optional).",
    )
    add_load_latest_argument(p)
    p.add_argument(
        "--target",
        default=None,
        help="Training vocabulary word for intervention (optional; default: base mode).",
    )
    p.add_argument(
        "--layer",
        type=int,
        default=13,
        help="Fallback layer if not in checkpoint config.",
    )
    p.add_argument(
        "--low-rank-dim",
        type=int,
        default=64,
        help="Fallback rank if not in checkpoint config.",
    )
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--greedy", action="store_true", help="Disable sampling.")
    p.add_argument(
        "--torch-dtype",
        default=None,
        choices=["bfloat16", "float16", "float32"],
    )
    p.add_argument(
        "--use-checkpoint-prompt",
        action="store_true",
        help="Always generate from the checkpoint's saved eval prompt.",
    )
    p.add_argument(
        "--check-smiles",
        action="store_true",
        help=(
            "After each decode, take the first whitespace-delimited token as a "
            "SMILES candidate and print (valid: true/false). When sanitization "
            "fails but the string still parses, append RDKit chemistry issues. "
            "When the candidate is invalid and SmiSelf is installed, also print "
            "the repaired SMILES."
        ),
    )
    p.add_argument(
        "--no-chat-template",
        action="store_true",
        help=(
            "Do not wrap prompts in the tokenizer chat template (base mode). "
            "Qwen base and chemistry CPT checkpoints often still have a "
            "chat_template; wrapping them as instruct models makes them "
            "repeat the user text."
        ),
    )
    p.add_argument(
        "--multi-turn",
        action="store_true",
        help="Start with history accumulation on (default: single-turn /no-chat).",
    )
    p.add_argument(
        "--interp-method",
        default="lerp",
        choices=sorted(VALID_INTERP_METHODS),
        help="Default bias interpolation method for /interpolate (lerp or slerp).",
    )
    args = p.parse_args()

    cache_dir = args.cache_dir or None
    do_sample = not args.greedy
    temperature = args.temperature
    accumulate_history = args.multi_turn
    interp_method = normalize_interp_method(args.interp_method)
    history: list[dict[str, str]] = []
    system_prompt: str | None = None
    cached_bias: np.ndarray | None = None
    intervention_label: str | None = None
    active_test_word: str | None = None
    bias_session_cache: dict[str, np.ndarray] = {}
    add_bias_network = False
    test_words: list[str] = []
    test_word_by_norm: dict[str, str] = {}
    test_interp: set[str] = set()
    definition_lookup: dict[str, str] | None = None
    encoder_extra_kwargs: dict = {}
    embed_task = "semantle"

    if args.checkpoint_dir:
        saved_cfg = load_merged_run_config(args.checkpoint_dir)
        embed_task = str(saved_cfg.get("task", "semantle"))
        model_name = saved_cfg.get("model_name", args.model_name)
        layer = saved_cfg.get("layer", args.layer)
        low_rank_dim = saved_cfg.get("low_rank_dim", args.low_rank_dim)
        position = saved_cfg.get("position", "l1")

        ckpt = load_eval_checkpoint(
            args.checkpoint_dir,
            model_name,
            layer,
            low_rank_dim,
            cache_dir,
            torch_dtype=args.torch_dtype,
            load_latest=bool(args.load_latest),
        )
        word_to_id = _index_words_by_id(ckpt.items)
        word_idx = None
        if args.target is not None:
            if args.target not in word_to_id:
                examples = ", ".join(list(word_to_id.keys())[:8])
                raise KeyError(
                    f"target {args.target!r} not in checkpoint vocab. Examples: {examples}"
                )
            word_idx = word_to_id[args.target]
            generation_mode: GenerationMode = "intervention"
        else:
            generation_mode = "base"

        has_checkpoint = True
        loaded_model_name = model_name
        use_chat_template = ckpt.from_chat_template
        system_prompt = system_prompt_from_cfg(ckpt.saved_cfg)
        add_bias_network = bool(saved_cfg.get("add_bias_network"))
        if add_bias_network:
            definition_lookup = definition_lookup_for_cfg(saved_cfg)
            # Encode the train vocabulary once and reuse for both the interp/extrap
            # test-set split and /learn's placement analysis (cached on ckpt).
            train_emb = get_train_embeddings(ckpt, task=embed_task)
            interp_set, extrap_set, _test_meta = build_test_sets(
                train_targets=ckpt.words,
                csv_paths=task_csv_paths(saved_cfg, task=embed_task),
                test_n_samples=int(
                    saved_cfg.get("test_n_samples", DEFAULT_TEST_N_SAMPLES)
                ),
                seed=int(saved_cfg.get("seed", 42)),
                pca_var=float(
                    saved_cfg.get("eval_bbox_pca_var", DEFAULT_BBOX_PCA_VAR)
                ),
                task=embed_task,
                train_embeddings=train_emb,
            )
            test_words = list(interp_set) + list(extrap_set)
            test_word_by_norm = {
                normalize_text(w, task=embed_task): w for w in test_words
            }
            test_interp = set(interp_set)
            encoder_extra_kwargs = bias_predict_kwargs(
                saved_cfg, tokenizer=ckpt.tokenizer
            )
        print(
            f"[repl] checkpoint={args.checkpoint_dir!r} "
            f"model={model_name!r} "
            f"mode={_mode_label(has_checkpoint=True, generation_mode=generation_mode, target=args.target, intervention_label=intervention_label)} "
            f"position={position!r} rank={low_rank_dim}"
        )
        if add_bias_network:
            enc_src = saved_cfg.get("bias_input_source", "embed_cache")
            if enc_src == "llm_encoder":
                print(
                    f"[repl] bias-network encoder: llm_encoder "
                    f"(definitions from {encoder_definitions_path(saved_cfg)!r})"
                )
            else:
                print("[repl] bias-network encoder: sentence_transformer")
            print(
                f"[repl] bias-network: /target-test-list, /target-test, /target-text "
                f"({len(test_words)} test words)"
            )
    else:
        ckpt = None
        word_to_id = {}
        word_idx = None
        low_rank_dim = args.low_rank_dim
        generation_mode = "base"
        has_checkpoint = False
        model, tokenizer, device, dtype_name, loaded_model_name = _load_base_model(
            args.model_name, cache_dir, args.torch_dtype
        )
        tokenizer_has_chat = bool(getattr(tokenizer, "chat_template", None))
        use_chat_template = tokenizer_has_chat and not args.no_chat_template
        print(
            f"[repl] model={loaded_model_name!r} cache_dir={cache_dir!r} "
            f"device={device} dtype={dtype_name}"
        )
        if tokenizer_has_chat and args.no_chat_template:
            print("[repl] tokenizer has a chat_template; wrapping disabled by --no-chat-template")
        elif tokenizer_has_chat:
            print(
                "[repl] note: tokenizer has a chat_template. Chemistry CPT "
                "weights are often completion models; if generations repeat "
                "the prompt, pass --no-chat-template and --greedy"
            )

    print(
        f"[repl] chat_template={'on' if use_chat_template else 'off'} "
        f"history={'on' if accumulate_history else 'off'} "
        f"sampling={_sampling_label(do_sample=do_sample, temperature=temperature, top_p=args.top_p)}"
        f"{' check_smiles=on' if args.check_smiles else ''}"
        f"{' system=on' if system_prompt else ''}"
    )
    print("Type a prompt and press Enter. Type /help for commands.")
    print()

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": args.top_p,
    }

    def _sync_gen_kwargs() -> None:
        gen_kwargs["do_sample"] = do_sample
        gen_kwargs["temperature"] = temperature

    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text in {"/quit", "/exit", "quit", "exit"}:
            break
        if user_text == "/help":
            _print_help(has_checkpoint=has_checkpoint, ckpt=ckpt)
            print()
            continue
        if user_text == "/clear":
            history.clear()
            print("[repl] history cleared")
            continue
        if user_text == "/zero":
            if not has_checkpoint:
                print("[repl] /zero only applies with --checkpoint-dir")
                continue
            generation_mode = "zero_bias"
            cached_bias = None
            intervention_label = None
            active_test_word = None
            word_idx = None
            print(
                f"[repl] mode={_mode_label(has_checkpoint=True, generation_mode=generation_mode, target=args.target, intervention_label=intervention_label)}"
            )
            continue
        if user_text == "/base":
            if not has_checkpoint:
                print("[repl] already in base mode")
                continue
            generation_mode = "base"
            cached_bias = None
            intervention_label = None
            active_test_word = None
            word_idx = None
            print(
                f"[repl] mode={_mode_label(has_checkpoint=True, generation_mode=generation_mode, target=args.target, intervention_label=intervention_label)}"
            )
            continue
        if user_text in {"/target-list", "/targets"}:
            if not has_checkpoint:
                print("[repl] /target-list only applies with --checkpoint-dir")
                continue
            _print_target_list(word_to_id)
            print()
            continue
        if user_text.startswith("/target "):
            if not has_checkpoint:
                print("[repl] /target only applies with --checkpoint-dir")
                continue
            new_target = user_text.split(maxsplit=1)[1].strip()
            if new_target not in word_to_id:
                print(f"[repl] unknown target {new_target!r}")
                continue
            args.target = new_target
            word_idx = word_to_id[new_target]
            cached_bias = None
            intervention_label = None
            active_test_word = None
            generation_mode = "intervention"
            print(
                f"[repl] mode={_mode_label(has_checkpoint=True, generation_mode=generation_mode, target=args.target, intervention_label=intervention_label)} "
                f"(id={word_idx})"
            )
            continue
        if user_text in {"/target-test-list", "/target-testlist"}:
            if not has_checkpoint:
                print("[repl] /target-test-list only applies with --checkpoint-dir")
                continue
            if not add_bias_network:
                print("[repl] /target-test-list requires a bias-network checkpoint")
                continue
            if not test_words:
                print("[repl] test set is empty (check semantle_csv in training config)")
                continue
            _print_target_test_list(test_words, interp_words=test_interp)
            print()
            continue
        if user_text.startswith("/target-test "):
            if not has_checkpoint:
                print("[repl] /target-test only applies with --checkpoint-dir")
                continue
            if not add_bias_network:
                print("[repl] /target-test requires a bias-network checkpoint")
                continue
            query = user_text.split(maxsplit=1)[1].strip()
            canonical = test_word_by_norm.get(
                normalize_text(query, task=embed_task)
            )
            if canonical is None:
                print(
                    f"[repl] {query!r} is not in the eval test set "
                    "(try /target-test-list)"
                )
                continue
            cache_key = (
                f"target-test:{normalize_text(canonical, task=embed_task)}"
            )
            if cache_key in bias_session_cache:
                bias_vec = bias_session_cache[cache_key]
            else:
                assert ckpt is not None
                bias_vec = predict_bias_vectors_for_words(
                    ckpt.reft_model,
                    [canonical],
                    definition_lookup=definition_lookup,
                    task=embed_task,
                    **encoder_extra_kwargs,
                )[0]
                bias_session_cache[cache_key] = bias_vec
            cached_bias = bias_vec
            word_idx = None
            active_test_word = canonical
            intervention_label = f"target-test={canonical!r}"
            generation_mode = "intervention"
            print(f"[repl] mode={_mode_label(has_checkpoint=True, generation_mode=generation_mode, target=None, intervention_label=intervention_label)}")
            continue
        if user_text.startswith("/target-text "):
            if not has_checkpoint:
                print("[repl] /target-text only applies with --checkpoint-dir")
                continue
            if not add_bias_network:
                print("[repl] /target-text requires a bias-network checkpoint")
                continue
            raw_text = user_text.split(maxsplit=1)[1]
            if not raw_text:
                print("[repl] usage: /target-text <text>")
                continue
            cache_key = f"target-text:{raw_text}"
            if cache_key in bias_session_cache:
                bias_vec = bias_session_cache[cache_key]
            else:
                assert ckpt is not None
                bias_vec = predict_bias_vectors_from_raw_texts(
                    ckpt.reft_model,
                    [raw_text],
                    task=embed_task,
                    **{
                        k: encoder_extra_kwargs[k]
                        for k in (
                            "tokenizer",
                            "encoder_max_length",
                            "encoder_layer_index",
                            "embed_model",
                        )
                        if k in encoder_extra_kwargs
                    },
                )[0]
                bias_session_cache[cache_key] = bias_vec
            cached_bias = bias_vec
            word_idx = None
            active_test_word = None
            intervention_label = f"target-text={raw_text!r}"
            generation_mode = "intervention"
            print(f"[repl] mode={_mode_label(has_checkpoint=True, generation_mode=generation_mode, target=None, intervention_label=intervention_label)}")
            continue
        if user_text.startswith("/learn "):
            if not has_checkpoint:
                print("[repl] /learn only applies with --checkpoint-dir")
                continue
            if not add_bias_network:
                print("[repl] /learn requires a bias-network checkpoint")
                continue
            body = user_text[len("/learn "):].strip()
            if "::" not in body:
                print("[repl] usage: /learn <target> :: <definition> [key=value ...]")
                continue
            segments = [seg.strip() for seg in body.split("::")]
            learn_target = segments[0]
            learn_definition = segments[1]
            learn_options = segments[2] if len(segments) >= 3 else ""
            if not learn_target or not learn_definition:
                print("[repl] usage: /learn <target> :: <definition> [key=value ...]")
                continue
            assert ckpt is not None
            learned_mu = _run_repl_learn(
                ckpt,
                checkpoint_dir=args.checkpoint_dir,
                target=learn_target,
                definition=learn_definition,
                options=learn_options,
            )
            if learned_mu is None:
                continue
            cached_bias = learned_mu
            word_idx = None
            active_test_word = learn_target
            intervention_label = f"learned={learn_target!r}"
            generation_mode = "intervention"
            print(
                f"[repl] mode={_mode_label(has_checkpoint=True, generation_mode=generation_mode, target=None, intervention_label=intervention_label)}"
            )
            continue
        if user_text in {"/learned-list", "/learnedlist"}:
            if not has_checkpoint:
                print("[repl] /learned-list only applies with --checkpoint-dir")
                continue
            records = load_learned_biases(args.checkpoint_dir)
            if not records:
                print("[repl] no learned biases persisted yet (use /learn)")
                continue
            print(f"[repl] {len(records)} learned bias vector(s):")
            for rec in records:
                flag = "ok" if rec.get("converged") else "max"
                print(
                    f"  {rec.get('target')!r}  sim={rec.get('final_sim', float('nan')):.3f}  "
                    f"decode={rec.get('decode')!r}  [{flag}]  {rec.get('learned_at', '')}"
                )
            print()
            continue
        if user_text.startswith("/learned "):
            if not has_checkpoint:
                print("[repl] /learned only applies with --checkpoint-dir")
                continue
            query = user_text.split(maxsplit=1)[1].strip()
            record = find_learned_bias(args.checkpoint_dir, query, task=embed_task)
            if record is None:
                print(f"[repl] no learned bias for {query!r} (try /learned-list)")
                continue
            cached_bias = np.asarray(record["mu"], dtype=np.float32)
            word_idx = None
            active_test_word = record.get("target")
            intervention_label = f"learned={record.get('target')!r}"
            generation_mode = "intervention"
            print(
                f"[repl] loaded learned bias for {record.get('target')!r} "
                f"(sim={record.get('final_sim', float('nan')):.3f}, decode={record.get('decode')!r})"
            )
            if record.get("classification") is not None or record.get("closest_train_word"):
                placement = record.get("classification") or "?"
                neighbor = record.get("closest_train_word")
                extra = ""
                if neighbor:
                    extra = f"  closest train={neighbor!r}"
                    if record.get("bias_distance") is not None:
                        extra += f" (bias L2={record['bias_distance']:.3f}"
                        if record.get("embed_sim_to_closest") is not None:
                            extra += f", embed sim={record['embed_sim_to_closest']:.3f}"
                        extra += ")"
                print(f"[repl] placement: {placement} w.r.t. train set{extra}")
            print(
                f"[repl] mode={_mode_label(has_checkpoint=True, generation_mode=generation_mode, target=None, intervention_label=intervention_label)}"
            )
            continue
        if user_text.startswith("/temp"):
            parts = user_text.split(maxsplit=1)
            if len(parts) == 1:
                print(
                    f"[repl] sampling={_sampling_label(do_sample=do_sample, temperature=temperature, top_p=args.top_p)}"
                )
                continue
            value = parts[1].strip().lower()
            if value == "off":
                do_sample = False
            else:
                try:
                    temperature = float(parts[1].strip())
                except ValueError:
                    print("[repl] usage: /temp <float>  or  /temp off")
                    continue
                if temperature <= 0:
                    print("[repl] temperature must be positive")
                    continue
                do_sample = True
            _sync_gen_kwargs()
            print(
                f"[repl] sampling={_sampling_label(do_sample=do_sample, temperature=temperature, top_p=args.top_p)}"
            )
            continue
        if user_text == "/chat-template" or user_text.startswith("/chat-template "):
            if has_checkpoint:
                print("[repl] /chat-template only applies in base mode (no --checkpoint-dir)")
                continue
            parts = user_text.split(maxsplit=1)
            if len(parts) == 1:
                print(f"[repl] chat_template={'on' if use_chat_template else 'off'}")
                continue
            value = parts[1].strip().lower()
            if value in {"on", "off"}:
                if value == "on" and not getattr(tokenizer, "chat_template", None):
                    print("[repl] tokenizer has no chat_template")
                    continue
                use_chat_template = value == "on"
                if not use_chat_template:
                    system_prompt = None
                print(f"[repl] chat_template={'on' if use_chat_template else 'off'}")
            else:
                print("[repl] usage: /chat-template on|off")
            continue
        if user_text == "/system" or user_text.startswith("/system "):
            parts = user_text.split(maxsplit=1)
            if len(parts) == 1:
                _print_system_prompt(system_prompt)
                continue
            raw = parts[1].strip()
            if raw.lower() == "off":
                system_prompt = None
                print("[repl] system=off")
                continue
            if raw in {"<<", '"""'}:
                end_tokens = (".", "/end")
                if raw == '"""':
                    end_tokens = (".", "/end", '"""')
                value = _read_multiline_block(end_tokens=end_tokens)
                if value is None:
                    print("[repl] system unchanged")
                    continue
                if not value:
                    print("[repl] empty system prompt; unchanged")
                    continue
            else:
                value = raw
            if not use_chat_template:
                print("[repl] /system only applies when chat_template is on")
                continue
            if args.use_checkpoint_prompt:
                print(
                    "[repl] /system is ignored while --use-checkpoint-prompt is set"
                )
            system_prompt = value
            _print_system_prompt(system_prompt)
            continue
        if user_text == "/chat":
            if accumulate_history:
                print("[repl] already accumulating history")
            else:
                accumulate_history = True
                history.clear()
                print("[repl] history accumulation on (history cleared)")
            continue
        if user_text in {"/no-chat", "/nochat"}:
            if not accumulate_history:
                print("[repl] already in single-turn mode")
            else:
                accumulate_history = False
                history.clear()
                print("[repl] single-turn mode (history cleared)")
            continue
        if user_text.startswith("/interp-method") or user_text.startswith(
            "/interp_method"
        ):
            parts = user_text.split(maxsplit=1)
            if len(parts) == 1:
                print(f"[repl] interp_method={interp_method}")
                continue
            try:
                interp_method = normalize_interp_method(parts[1].strip())
            except ValueError as exc:
                print(f"[repl] {exc}")
                continue
            print(f"[repl] interp_method={interp_method}")
            continue
        if user_text.startswith("/interpolate"):
            if not has_checkpoint:
                print("[repl] /interpolate only applies with --checkpoint-dir")
                continue
            parts = user_text.split()
            method_for_run = interp_method
            start = 1
            if len(parts) >= 2 and parts[1].lower() in VALID_INTERP_METHODS:
                method_for_run = normalize_interp_method(parts[1])
                start = 2
            tail_len = len(parts) - start
            if tail_len not in (3, 4):
                print(
                    "[repl] usage: /interpolate [lerp|slerp] <target1> <target2> "
                    "<n_steps> [<n_samples>]"
                )
                continue
            w1, w2, n_steps_str = parts[start : start + 3]
            n_samples_str = parts[start + 3] if tail_len == 4 else "1"
            try:
                n_steps = int(n_steps_str)
                n_samples = int(n_samples_str)
            except ValueError:
                print(
                    "[repl] usage: /interpolate [lerp|slerp] <target1> <target2> "
                    "<n_steps> [<n_samples>]"
                )
                continue
            if n_steps < 2:
                print("[repl] n_steps must be >= 2")
                continue
            if n_samples < 1:
                print("[repl] n_samples must be >= 1")
                continue
            assert ckpt is not None
            _run_repl_interpolation(
                ckpt,
                word1=w1,
                word2=w2,
                n_steps=n_steps,
                n_samples=n_samples,
                word_to_id=word_to_id,
                max_new_tokens=gen_kwargs["max_new_tokens"],
                do_sample=gen_kwargs["do_sample"],
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                interp_method=method_for_run,
            )
            continue

        if has_checkpoint:
            assert ckpt is not None
            prompt, from_chat, content_span = _build_checkpoint_prompt(
                ckpt,
                user_text=user_text,
                history=history,
                accumulate_history=accumulate_history,
                use_checkpoint_prompt=args.use_checkpoint_prompt,
                generation_mode=generation_mode,
                model_name=loaded_model_name,
                system_prompt=system_prompt,
            )

            if generation_mode == "base":
                text = _generate_base(
                    ckpt.reft_model.model,
                    ckpt.tokenizer,
                    prompt,
                    from_chat_template=from_chat,
                    assistant_suffix=ckpt.assistant_suffix,
                    **gen_kwargs,
                )
            elif generation_mode == "zero_bias":
                zero_bias = [0.0] * low_rank_dim
                text = generate_text(
                    ckpt.reft_model,
                    ckpt.tokenizer,
                    prompt,
                    zero_bias,
                    use_sample=do_sample,
                    temperature=temperature,
                    top_p=args.top_p,
                    position=ckpt.saved_cfg.get("position", "l1"),
                    assistant_suffix=ckpt.assistant_suffix,
                    from_chat_template=from_chat,
                    intervention_token_id=ckpt.intervention_token_id,
                    content_span=content_span,
                )
            else:
                if cached_bias is not None:
                    subspace = cached_bias
                elif word_idx is not None:
                    subspace = word_idx
                else:
                    print(
                        "[repl] no intervention target; use /target, /target-test, "
                        "or /target-text"
                    )
                    continue
                text = generate_text(
                    ckpt.reft_model,
                    ckpt.tokenizer,
                    prompt,
                    subspace,
                    use_sample=do_sample,
                    temperature=temperature,
                    top_p=args.top_p,
                    position=ckpt.saved_cfg.get("position", "l1"),
                    assistant_suffix=ckpt.assistant_suffix,
                    from_chat_template=from_chat,
                    intervention_token_id=ckpt.intervention_token_id,
                    content_span=content_span,
                )

            if (
                not args.use_checkpoint_prompt
                and accumulate_history
            ):
                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": text})
        else:
            prompt, from_chat = _build_base_prompt(
                tokenizer,
                user_text,
                history,
                use_chat_template=use_chat_template,
                accumulate_history=accumulate_history,
                system_prompt=system_prompt,
            )

            text = _generate_base(
                model,
                tokenizer,
                prompt,
                from_chat_template=from_chat,
                **gen_kwargs,
            )
            if accumulate_history:
                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": text})

        print(f"bot> {text}")
        if args.check_smiles:
            print(_smiles_status_line(text))
        if has_checkpoint and generation_mode == "intervention":
            seen, is_target = _target_intervention_flags(
                text,
                active_test_word if active_test_word is not None else args.target,
                word_to_id,
            )
            print(f"[SEEN={seen}; IS_TARGET={is_target}]")
        print()


if __name__ == "__main__":
    main()
