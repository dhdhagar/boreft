import math

import torch
import torch.nn.functional as F

# Shared log-variance bounds for VAE sampling and KL (keep in sync with interventions).
LOGVAR_MIN = -14.0
LOGVAR_MAX = 10.0


def clamp_logvar(logvar: torch.Tensor) -> torch.Tensor:
    """Clamp log-variance to the stable range used for sampling and KL."""
    return torch.clamp(logvar, min=LOGVAR_MIN, max=LOGVAR_MAX)


def kl_divergence(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    prior_var: float = 1.0,
    free_bits: float = 0.0,
) -> torch.Tensor:
    """KL divergence from N(mu, exp(logvar)) to N(0, prior_var), averaged over the batch.

    ``logvar`` is clamped to [LOGVAR_MIN, LOGVAR_MAX] so KL matches the distribution
    used in the reparameterization trick.

    With prior_var=1 this reduces to the standard VAE KL:
        -0.5 * sum(1 + logvar - mu² - exp(logvar))

    With prior_var=tau²:
        -0.5 * sum(1 + logvar - log(tau²) - (mu² + exp(logvar)) / tau²)

    ``free_bits`` > 0 applies the free-bits floor (Kingma et al., 2016): each latent
    dimension's batch-averaged KL is clamped to be at least ``free_bits`` nats before
    summing over dimensions, so KL below the floor contributes no gradient. This
    reserves capacity in every dimension and mitigates posterior collapse. The clamp
    is applied to the batch-averaged per-dim KL (not per example) so a dimension that
    is informative on average is never penalized for occasional low-KL examples.
    """
    if prior_var <= 0:
        raise ValueError(f"prior_var must be positive, got {prior_var}")
    if free_bits < 0:
        raise ValueError(f"free_bits must be non-negative, got {free_bits}")
    logvar = clamp_logvar(logvar)
    log_prior_var = math.log(prior_var)
    # Per-dimension KL [batch, dim]
    kl_per_dim = -0.5 * (
        1 + logvar - log_prior_var - (mu.pow(2) + logvar.exp()) / prior_var
    )
    if free_bits > 0.0:
        # Clamp batch-averaged per-dim KL at the floor, then sum over dims.
        kl_dim = torch.clamp(kl_per_dim.mean(dim=0), min=free_bits)
        return kl_dim.sum()
    return kl_per_dim.sum(dim=-1).mean()


# Loss coefficients that may appear in ``--linear-annealing-map``.
LINEAR_ANNEAL_KEYS = frozenset({"kl_beta", "lambda_ce", "lambda_sdpo"})

# Serializable map entry: (start, end, epochs).
LinearAnnealEntry = tuple[float, float, int]
LinearAnnealMap = dict[str, LinearAnnealEntry]


def linear_anneal(
    start: float,
    end: float,
    *,
    epochs: int,
    global_step: int,
    steps_per_epoch: int,
) -> float:
    """Linear ramp from ``start`` to ``end`` over ``epochs`` * ``steps_per_epoch``."""
    anneal_steps = max(1, int(epochs) * max(1, int(steps_per_epoch)))
    t = min(1.0, float(global_step) / float(anneal_steps))
    return float(start) + t * (float(end) - float(start))


def effective_loss_coeffs(
    defaults: dict[str, float],
    anneal_map: LinearAnnealMap | None,
    *,
    global_step: int,
    steps_per_epoch: int,
) -> dict[str, float]:
    """Return loss coefficients with any linear-annealing-map entries applied."""
    out = {k: float(v) for k, v in defaults.items()}
    if not anneal_map:
        return out
    for key, (start, end, epochs) in anneal_map.items():
        out[key] = linear_anneal(
            start,
            end,
            epochs=epochs,
            global_step=global_step,
            steps_per_epoch=steps_per_epoch,
        )
    return out


def _parse_anneal_epochs(text: str, *, entry: str) -> int:
    """Parse a positive integer epoch count (accepts ``50`` or ``50.0``)."""
    try:
        value = float(text)
    except ValueError as e:
        raise ValueError(
            f"invalid --linear-annealing-map entry {entry!r}; "
            "epochs must be a positive integer"
        ) from e
    epochs = int(value)
    if abs(value - epochs) > 1e-9:
        raise ValueError(
            f"invalid --linear-annealing-map entry {entry!r}; "
            "epochs must be a whole number"
        )
    if epochs <= 0:
        raise ValueError(
            f"invalid --linear-annealing-map entry {entry!r}; "
            f"epochs must be > 0 (got {epochs})"
        )
    return epochs


def parse_linear_annealing_map(
    raw: str | None,
    *,
    defaults: dict[str, float],
) -> LinearAnnealMap:
    """Parse ``KEY=START:END:EPOCHS`` or ``KEY=START:EPOCHS`` (end from defaults).

    Entries are comma-separated. Allowed keys: ``kl_beta``, ``lambda_ce``,
    ``lambda_sdpo``. Shorthand ``KEY=START:EPOCHS`` uses ``defaults[KEY]`` as end.
    """
    if raw is None:
        return {}
    text = str(raw).strip()
    if not text:
        return {}

    out: LinearAnnealMap = {}
    for part in text.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(
                f"invalid --linear-annealing-map entry {chunk!r}; "
                "expected KEY=START:END:EPOCHS or KEY=START:EPOCHS"
            )
        key, _, spec = chunk.partition("=")
        key = key.strip()
        spec = spec.strip()
        if key not in LINEAR_ANNEAL_KEYS:
            raise ValueError(
                f"unknown --linear-annealing-map key {key!r}; "
                f"allowed: {sorted(LINEAR_ANNEAL_KEYS)}"
            )
        if key in out:
            raise ValueError(f"duplicate --linear-annealing-map key {key!r}")
        pieces = [p.strip() for p in spec.split(":")]
        if len(pieces) == 2:
            if key not in defaults:
                raise ValueError(
                    f"--linear-annealing-map shorthand for {key!r} requires a "
                    f"configured default end value"
                )
            try:
                start = float(pieces[0])
            except ValueError as e:
                raise ValueError(
                    f"invalid --linear-annealing-map entry {chunk!r}; "
                    "shorthand is KEY=START:EPOCHS"
                ) from e
            # If the second field looks like a non-integer coefficient, the user
            # likely omitted epochs (KEY=START:END instead of START:END:EPOCHS).
            try:
                epochs = _parse_anneal_epochs(pieces[1], entry=chunk)
            except ValueError as e:
                raise ValueError(
                    f"invalid --linear-annealing-map entry {chunk!r}; "
                    "shorthand is KEY=START:EPOCHS (for KEY=START:END:EPOCHS "
                    "provide three fields)"
                ) from e
            end = float(defaults[key])
        elif len(pieces) == 3:
            try:
                start = float(pieces[0])
                end = float(pieces[1])
            except ValueError as e:
                raise ValueError(
                    f"invalid --linear-annealing-map entry {chunk!r}; "
                    "expected KEY=START:END:EPOCHS"
                ) from e
            epochs = _parse_anneal_epochs(pieces[2], entry=chunk)
        else:
            raise ValueError(
                f"invalid --linear-annealing-map entry {chunk!r}; "
                "expected KEY=START:END:EPOCHS or KEY=START:EPOCHS"
            )
        if start < 0 or end < 0:
            raise ValueError(
                f"--linear-annealing-map {key} start/end must be >= 0 "
                f"(got start={start}, end={end})"
            )
        out[key] = (start, end, epochs)
    return out


def serialize_linear_annealing_map(
    anneal_map: LinearAnnealMap | None,
) -> dict[str, list[float | int]] | None:
    """JSON-friendly form ``{key: [start, end, epochs]}`` (or ``None`` if empty)."""
    if not anneal_map:
        return None
    return {
        key: [float(start), float(end), int(epochs)]
        for key, (start, end, epochs) in sorted(anneal_map.items())
    }


def load_linear_annealing_map(
    raw: dict | str | None,
) -> LinearAnnealMap:
    """Load a map from config JSON, a CLI string, or an already-parsed dict."""
    if raw is None:
        return {}
    if isinstance(raw, str):
        # String form without shorthand defaults: require START:END:EPOCHS.
        return parse_linear_annealing_map(raw, defaults={})
    if isinstance(raw, dict):
        out: LinearAnnealMap = {}
        for key, value in raw.items():
            key_s = str(key)
            if key_s not in LINEAR_ANNEAL_KEYS:
                raise ValueError(
                    f"unknown linear_annealing_map key {key_s!r}; "
                    f"allowed: {sorted(LINEAR_ANNEAL_KEYS)}"
                )
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                raise ValueError(
                    f"linear_annealing_map[{key_s!r}] must be [start, end, epochs]"
                )
            start, end = float(value[0]), float(value[1])
            epochs = _parse_anneal_epochs(str(value[2]), entry=f"{key_s}={value!r}")
            if start < 0 or end < 0:
                raise ValueError(
                    f"linear_annealing_map[{key_s!r}] start/end must be >= 0"
                )
            out[key_s] = (start, end, epochs)
        return out
    raise ValueError(
        f"linear_annealing_map must be a dict or string (got {type(raw).__name__})"
    )


def align_linear_annealing_map(
    anneal_map: LinearAnnealMap | None,
    *,
    kl_beta: float,
    lambda_ce: float,
    lambda_sdpo: float,
    override_ends: set[str] | frozenset[str] | None = None,
) -> LinearAnnealMap:
    """Align a checkpoint/run annealing map to this run's loss coefficients.

    - Drop anneals for terms whose static coefficient is ``<= 0`` (disabled).
    - For keys in ``override_ends``, rewrite the anneal *end* to the static
      coefficient (so an explicit ``--lambda-ce`` / recovery override retargets
      the schedule).
    - Otherwise keep the persisted ``(start, end, epochs)`` so cool-downs
      (high→low) survive inherit when the static flag is the training gate
      rather than the anneal end.
    """
    if not anneal_map:
        return {}
    statics = {
        "kl_beta": float(kl_beta),
        "lambda_ce": float(lambda_ce),
        "lambda_sdpo": float(lambda_sdpo),
    }
    rewrite = set(override_ends or ())
    out: LinearAnnealMap = {}
    for key, (start, end, epochs) in anneal_map.items():
        if key not in statics:
            continue
        target = statics[key]
        # Disabled terms should stay at the static 0 from defaults, not ramp from
        # a checkpoint start value.
        if target <= 0.0:
            continue
        aligned_end = target if key in rewrite else float(end)
        out[key] = (float(start), aligned_end, int(epochs))
    return out


def resolve_linear_annealing_map(
    *,
    raw_map: str | dict | None,
    defaults: dict[str, float],
) -> LinearAnnealMap:
    """Parse ``raw_map`` into a normalized annealing map."""
    if isinstance(raw_map, dict):
        return load_linear_annealing_map(raw_map)
    return parse_linear_annealing_map(raw_map, defaults=defaults)


def legacy_kl_annealing_map_from_saved_cfg(
    saved_cfg: dict,
    *,
    kl_beta: float,
) -> LinearAnnealMap:
    """Synthesize a ``kl_beta`` anneal entry from pre-map checkpoint fields.

    Older checkpoints may store ``kl_beta_anneal`` / ``kl_beta_start`` /
    ``kl_beta_anneal_epochs`` instead of ``linear_annealing_map``.
    """
    if not bool(saved_cfg.get("kl_beta_anneal", False)):
        return {}
    epochs = int(saved_cfg.get("kl_beta_anneal_epochs") or 0)
    if epochs <= 0:
        return {}
    start = float(saved_cfg.get("kl_beta_start", 0.0))
    return {"kl_beta": (start, float(kl_beta), epochs)}


def annealing_map_from_saved_cfg(
    saved_cfg: dict,
    *,
    kl_beta: float | None = None,
) -> LinearAnnealMap:
    """Load a checkpoint annealing map (persisted map or legacy KL fields).

    If ``linear_annealing_map`` is present (even as ``null`` / empty), that wins
    and legacy ``kl_beta_anneal*`` fields are ignored — so a recovery that
    disables annealing can persist the disable.
    """
    if "linear_annealing_map" in saved_cfg:
        raw = saved_cfg["linear_annealing_map"]
        if not raw:
            return {}
        return load_linear_annealing_map(raw)
    beta = float(
        kl_beta
        if kl_beta is not None
        else (saved_cfg.get("kl_beta", 0.0) or 0.0)
    )
    return legacy_kl_annealing_map_from_saved_cfg(saved_cfg, kl_beta=beta)


def format_linear_annealing_map_cli(
    anneal_map: LinearAnnealMap | None,
) -> str:
    """CLI-style ``KEY=START:END:EPOCHS`` string (empty when no anneals)."""
    if not anneal_map:
        return ""
    parts: list[str] = []
    for key, (start, end, epochs) in sorted(anneal_map.items()):
        parts.append(f"{key}={float(start):g}:{float(end):g}:{int(epochs)}")
    return ",".join(parts)


def l2_reg(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """L2 regularization on the mu vectors.
    Pulls the per-word means toward the origin without touching variance.
    """
    # Compute L2 norm squared per vector, then mean over batch
    return mu.pow(2).sum(dim=-1).mean()


def none_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """No auxiliary loss — pure cross-entropy from the language model only."""
    return torch.tensor(0.0, device=mu.device)


def manifold_matching_loss(
    z: torch.Tensor,
    ref: torch.Tensor,
    exclude_diagonal: bool = True,
) -> torch.Tensor:
    """Match pairwise cosine-distance structure of z to reference embeddings.

    mm_loss = MSE(Dz, Dr) with Dz = 1 - z_norm @ z_norm.T, Dr = 1 - ref_norm @ ref_norm.T.
    Optionally excludes diagonal entries.

    z   : [B, d]  learned batch codes (e.g. VAE mu or linear word_bias rows).
    ref : [B, m]  gathered ref[word_ids] (e.g. sentence-transformer emb(target)).
    """
    B = z.size(0)
    if exclude_diagonal and B < 2:
        return z.new_tensor(0.0)

    z_n = F.normalize(z.float(), dim=-1, eps=1e-12)
    r_n = F.normalize(ref.float(), dim=-1, eps=1e-12)

    Dz = 1.0 - z_n @ z_n.T
    Dr = 1.0 - r_n @ r_n.T

    if exclude_diagonal:
        mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
        return F.mse_loss(Dz[mask], Dr[mask])
    return F.mse_loss(Dz, Dr)


def manifold_matching_loss_topk_anchors(
    embed_cache: torch.Tensor,
    z_full: torch.Tensor,
    batch_word_ids: torch.Tensor,
    k: int,
    t: int,
    exclude_diagonal: bool = True,
) -> torch.Tensor:
    """Manifold matching on per-word local neighborhoods in reference space.

    For **every** batch word, build a loss subset of ``{word} ∪ {k nearest ref neighbours}``.
    Optionally, ``t > 0`` randomly sampled batch words are also appended to each word's subset
    to introduce batch-level noise/interactions (same as the old batch-wise path).

    Args:
        embed_cache:    [N, m] full reference embeddings (vocabulary order).
        z_full:         [N, d] full code table (word_mu.weight or word_bias.weight).
        batch_word_ids: [B] word ids in the current minibatch.
        k:              k nearest neighbours in reference space per word. ``k <= 0`` falls
                        back to the plain batch-wise MM loss.
        t:              Extra random batch words appended to each word's subset (0 = none).
        exclude_diagonal: Passed to :func:`manifold_matching_loss`.

    Returns:
        Scalar mean MM loss over batch words. Falls back to batch-only MM when ``k <= 0``.
    """
    device = embed_cache.device
    B = int(batch_word_ids.numel())
    if B < 1:
        return embed_cache.new_tensor(0.0)

    # k=0: plain batch-wise MM (original behaviour)
    if k <= 0:
        return manifold_matching_loss(
            z_full[batch_word_ids],
            embed_cache[batch_word_ids],
            exclude_diagonal=exclude_diagonal,
        )

    N = embed_cache.size(0)
    k_eff = min(int(k), max(0, N - 1))
    if k_eff < 1:
        return manifold_matching_loss(
            z_full[batch_word_ids],
            embed_cache[batch_word_ids],
            exclude_diagonal=exclude_diagonal,
        )

    # Precompute k nearest ref neighbours for every batch word  [B, k_eff]
    ref_n = F.normalize(embed_cache.float(), dim=-1, eps=1e-12)
    sim = ref_n[batch_word_ids] @ ref_n.T  # [B, N]
    sim[torch.arange(B, device=device), batch_word_ids] = float("-inf")
    _, neigh = torch.topk(sim, k=k_eff, dim=1)  # [B, k_eff]

    # Optional: t random batch words to add to each word's subset
    t_extra = max(0, min(int(t), B - 1))

    losses = []
    for i in range(B):
        core_ids = torch.cat([batch_word_ids[i].view(1), neigh[i]])  # 1 + k_eff
        if t_extra > 0:
            # Sample t_extra words from the rest of the batch (excluding position i)
            other = torch.cat([batch_word_ids[:i], batch_word_ids[i + 1 :]])
            perm = torch.randperm(other.numel(), device=device)
            core_ids = torch.cat([core_ids, other[perm[:t_extra]]])
        ids = torch.unique(core_ids)
        losses.append(
            manifold_matching_loss(
                z_full[ids], embed_cache[ids], exclude_diagonal=exclude_diagonal
            )
        )
    return torch.stack(losses).mean()


def semantic_ranking_loss_topk_anchors(
    embed_cache: torch.Tensor,
    z_full: torch.Tensor,
    batch_word_ids: torch.Tensor,
    k: int,
    t: int,
    temperature: float = 0.1,
    exclude_diagonal: bool = True,
    min_ref_gap: float = 0.0,
) -> torch.Tensor:
    """Semantic ranking loss on per-word local neighborhoods in reference space.

    For **every** batch word, build a loss subset of ``{word} ∪ {k nearest ref neighbours}``.
    Optionally, ``t > 0`` randomly sampled batch words are also appended to each word's subset
    to introduce batch-level noise/interactions (same as the old batch-wise path).

    Args:
        embed_cache:    [N, m] full reference embeddings (vocabulary order).
        z_full:         [N, d] full code table (word_mu.weight or word_bias.weight).
        batch_word_ids: [B] word ids in the current minibatch.
        k:              k nearest neighbours in reference space per word. ``k <= 0`` falls
                        back to plain batch-wise ranking loss.
        t:              Extra random batch words appended to each word's subset (0 = none).
        temperature:    Softmax temperature for the ranking loss.
        exclude_diagonal: Passed to :func:`semantic_ranking_loss`.
        min_ref_gap:    Minimum ref-similarity gap between consecutive ranking steps.

    Returns:
        Scalar mean ranking loss over batch words. Falls back to batch-only ranking when
        ``k <= 0``.
    """
    device = embed_cache.device
    B = int(batch_word_ids.numel())
    if B < 1:
        return embed_cache.new_tensor(0.0)

    # k=0: plain batch-wise ranking (original behaviour)
    if k <= 0:
        return semantic_ranking_loss(
            z_full[batch_word_ids],
            embed_cache[batch_word_ids],
            temperature=temperature,
            exclude_diagonal=exclude_diagonal,
            min_ref_gap=min_ref_gap,
        )

    N = embed_cache.size(0)
    k_eff = min(int(k), max(0, N - 1))
    if k_eff < 1:
        return semantic_ranking_loss(
            z_full[batch_word_ids],
            embed_cache[batch_word_ids],
            temperature=temperature,
            exclude_diagonal=exclude_diagonal,
            min_ref_gap=min_ref_gap,
        )

    # Precompute k nearest ref neighbours for every batch word  [B, k_eff]
    ref_n = F.normalize(embed_cache.float(), dim=-1, eps=1e-12)
    sim = ref_n[batch_word_ids] @ ref_n.T  # [B, N]
    sim[torch.arange(B, device=device), batch_word_ids] = float("-inf")
    _, neigh = torch.topk(sim, k=k_eff, dim=1)  # [B, k_eff]

    # Optional: t random batch words to add to each word's subset
    t_extra = max(0, min(int(t), B - 1))

    losses = []
    for i in range(B):
        core_ids = torch.cat([batch_word_ids[i].view(1), neigh[i]])  # 1 + k_eff
        if t_extra > 0:
            # Sample t_extra words from the rest of the batch (excluding position i)
            other = torch.cat([batch_word_ids[:i], batch_word_ids[i + 1 :]])
            perm = torch.randperm(other.numel(), device=device)
            core_ids = torch.cat([core_ids, other[perm[:t_extra]]])
        ids = torch.unique(core_ids)
        losses.append(
            semantic_ranking_loss(
                z_full[ids],
                embed_cache[ids],
                temperature=temperature,
                exclude_diagonal=exclude_diagonal,
                min_ref_gap=min_ref_gap,
            )
        )
    return torch.stack(losses).mean()


def semantic_ranking_loss(
    z: torch.Tensor,
    ref: torch.Tensor,
    temperature: float = 0.1,
    exclude_diagonal: bool = True,
    min_ref_gap: float = 0.0,
) -> torch.Tensor:
    """PRO / Plackett-Luce style batch-wise ranking loss (fully vectorized).

    For each anchor word i, the batch candidates are sorted by descending
    reference similarity.  The loss encourages the learned-space similarities
    to respect that same ordering via recursive InfoNCE contrasts:

        L = - mean_k  log [ exp(Sz[i,pi_k] / T)
                            / sum_{m>=k} exp(Sz[i,pi_m] / T) ]

    averaged over all anchors.

    Implementation uses the suffix-logsumexp identity to eliminate Python
    loops entirely:
        logsumexp(s[k:])  =  flip( logcumsumexp( flip(s) ) )[k]

    Args:
        z   : [B, d]  learned batch codes (word_mu rows or word_bias rows).
        ref : [B, m]  reference embeddings for the same batch words.
        temperature: softmax temperature T (lower = sharper ranking).
        exclude_diagonal: ignore each word's similarity to itself.
        min_ref_gap: skip ranking steps where consecutive reference
            similarities differ by less than this value.  0.0 = no filtering.
    """
    B = z.size(0)
    if B < 3 if exclude_diagonal else B < 2:
        return z.new_tensor(0.0)

    z_n = F.normalize(z.float(), dim=-1, eps=1e-12)
    r_n = F.normalize(ref.float(), dim=-1, eps=1e-12)

    Sz = z_n @ z_n.T  # [B, B] learned cosine similarities
    Sr = r_n @ r_n.T  # [B, B] reference cosine similarities

    if exclude_diagonal:
        eye = torch.eye(B, dtype=torch.bool, device=z.device)
        # Push self-scores to -inf so they sort last in Sr and are excluded
        Sz = Sz.masked_fill(eye, float("-inf"))
        Sr = Sr.masked_fill(eye, float("-inf"))

    # Sort ALL candidates per anchor by descending reference similarity.
    # order[i, k] = column index of the k-th closest reference neighbour of i.
    order = torch.argsort(Sr, dim=1, descending=True)  # [B, B]
    Sz_sorted = torch.gather(Sz, 1, order)  # [B, B] scores in ref rank order
    Sr_sorted = torch.gather(Sr, 1, order)  # [B, B] ref sims in descending order

    if exclude_diagonal:
        # Last column is the self-element (-inf); drop it.
        Sz_sorted = Sz_sorted[:, :-1]  # [B, K]  K = B-1
        Sr_sorted = Sr_sorted[:, :-1]  # [B, K]

    K = Sz_sorted.shape[1]
    if K < 2:
        return z.new_tensor(0.0)

    scores_T = Sz_sorted / temperature  # [B, K]

    # Suffix logsumexp: lse[i, k] = logsumexp(scores_T[i, k:])
    # = flip over k of logcumsumexp of the flipped sequence.
    lse = torch.logcumsumexp(scores_T.flip(1), dim=1).flip(1)  # [B, K]

    # log P(rank-k candidate wins from its suffix) = scores_T[i,k] - lse[i,k]
    log_probs = scores_T - lse  # [B, K]

    # Only the first K-1 steps are meaningful (last has a trivial 1-item suffix).
    step_lp = log_probs[:, :-1]  # [B, K-1]
    step_mask = torch.ones(B, K - 1, device=z.device)  # [B, K-1]

    if min_ref_gap > 0.0:
        # Gap between consecutive reference sims; skip steps with gap < threshold.
        gaps = Sr_sorted[:, :-1] - Sr_sorted[:, 1:]  # [B, K-1]
        step_mask = (gaps >= min_ref_gap).float()

    # Per-anchor loss = -mean over valid steps
    n_valid = step_mask.sum(dim=1).clamp(min=1)  # [B]
    anchor_loss = -(step_lp * step_mask).sum(dim=1) / n_valid  # [B]

    # Only average over anchors that had at least one valid step
    has_steps = step_mask.sum(dim=1) > 0
    if not has_steps.any():
        return z.new_tensor(0.0)
    return anchor_loss[has_steps].mean()


SDPO_DIVERGENCES = ("forward_kl", "reverse_kl", "js")
_TOPK_TAIL_EPS = 1e-8


def _divergence_from_logp(
    student_logp: torch.Tensor,
    teacher_logp: torch.Tensor,
    divergence: str,
) -> torch.Tensor:
    """Per-row divergence on a last-dim simplex given as log-probabilities."""
    if divergence == "forward_kl":
        return (teacher_logp.exp() * (teacher_logp - student_logp)).sum(dim=-1)
    if divergence == "reverse_kl":
        return (student_logp.exp() * (student_logp - teacher_logp)).sum(dim=-1)
    if divergence == "js":
        mix_logp = torch.logsumexp(
            torch.stack([student_logp, teacher_logp], dim=0) - math.log(2.0), dim=0
        )
        kl_s_m = (student_logp.exp() * (student_logp - mix_logp)).sum(dim=-1)
        kl_t_m = (teacher_logp.exp() * (teacher_logp - mix_logp)).sum(dim=-1)
        return 0.5 * (kl_s_m + kl_t_m)
    raise ValueError(
        f"divergence must be one of {SDPO_DIVERGENCES}, got {divergence!r}"
    )


def _kl_mass_logp(
    source_mass: torch.Tensor,
    source_logp: torch.Tensor,
    target_logp: torch.Tensor,
) -> torch.Tensor:
    """``sum source_mass * (source_logp - target_logp)``; 0-mass bins contribute 0."""
    return (source_mass * (source_logp - target_logp)).sum(dim=-1)


def _divergence_from_mass_logp(
    student_mass: torch.Tensor,
    student_logp: torch.Tensor,
    teacher_mass: torch.Tensor,
    teacher_logp: torch.Tensor,
    divergence: str,
) -> torch.Tensor:
    if divergence == "forward_kl":
        return _kl_mass_logp(teacher_mass, teacher_logp, student_logp)
    if divergence == "reverse_kl":
        return _kl_mass_logp(student_mass, student_logp, teacher_logp)
    if divergence == "js":
        mix_mass = 0.5 * (student_mass + teacher_mass)
        mix_logp = torch.log(mix_mass.clamp(min=_TOPK_TAIL_EPS))
        return 0.5 * (
            _kl_mass_logp(student_mass, student_logp, mix_logp)
            + _kl_mass_logp(teacher_mass, teacher_logp, mix_logp)
        )
    raise ValueError(
        f"divergence must be one of {SDPO_DIVERGENCES}, got {divergence!r}"
    )


def _topk_mass_logp(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    topk: int,
    add_tail: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Student-ranked top-K masses and log-probs, optionally with a tail bucket.

    Masses are true softmax values (via ``logsumexp`` over V), not a softmax
    renormalized over K. The teacher is detached before any reduction.
    """
    teacher_logits = teacher_logits.detach()
    vocab = student_logits.shape[-1]
    k = min(int(topk), vocab)
    student_logz = torch.logsumexp(student_logits, dim=-1, keepdim=True)
    teacher_logz = torch.logsumexp(teacher_logits, dim=-1, keepdim=True)
    _, indices = torch.topk(student_logits, k, dim=-1)
    student_logp = student_logits.gather(-1, indices) - student_logz
    teacher_logp = teacher_logits.gather(-1, indices) - teacher_logz
    student_mass = student_logp.exp()
    teacher_mass = teacher_logp.exp()
    if not add_tail or k >= vocab:
        return student_mass, student_logp, teacher_mass, teacher_logp
    student_tail = (1.0 - student_mass.sum(dim=-1, keepdim=True)).clamp(min=0.0)
    teacher_tail = (1.0 - teacher_mass.sum(dim=-1, keepdim=True)).clamp(min=0.0)
    student_log_tail = torch.log(student_tail.clamp(min=_TOPK_TAIL_EPS))
    teacher_log_tail = torch.log(teacher_tail.clamp(min=_TOPK_TAIL_EPS))
    return (
        torch.cat([student_mass, student_tail], dim=-1),
        torch.cat([student_logp, student_log_tail], dim=-1),
        torch.cat([teacher_mass, teacher_tail], dim=-1),
        torch.cat([teacher_logp, teacher_log_tail], dim=-1),
    )


def sdpo_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    divergence: str = "forward_kl",
    topk: int | None = None,
    add_tail: bool = True,
) -> torch.Tensor:
    """Per-token logit-distillation loss between a student and a (stopgrad) teacher.

    Both tensors are ``[N, V]`` logits gathered at aligned next-token positions
    (``N`` = total target tokens across the batch, ``V`` = vocab). The teacher is
    always detached, so gradients flow only into ``student_logits`` (and thus into
    the intervention parameters that produced them). Returns the mean over the
    ``N`` positions (token-mean), scaled by ``temperature**2`` so the gradient
    magnitude is comparable across temperatures (standard KD rescaling).

    ``topk`` is the SDPO A.3 approximation: KL over the student's top-K tokens
    plus a tail bucket for leftover mass. ``None`` or ``0`` keeps the full vocab.
    Hübotter TTT uses ``topk=20``.

    divergence:
        "forward_kl"  — KL(teacher || student), mass-covering (classic KD).
        "reverse_kl"  — KL(student || teacher), mode-seeking (SDPO Eq. 1).
        "js"          — symmetric Jensen-Shannon (more stable; SDPO §2.3).
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student/teacher logits shape mismatch: "
            f"{tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
        )
    if student_logits.numel() == 0:
        # No target tokens this batch — return a differentiable zero.
        return student_logits.float().sum() * 0.0
    if divergence not in SDPO_DIVERGENCES:
        raise ValueError(
            f"divergence must be one of {SDPO_DIVERGENCES}, got {divergence!r}"
        )
    if topk is not None and topk < 0:
        raise ValueError(f"topk must be nonnegative or None, got {topk}")

    student = student_logits.float() / temperature
    teacher = teacher_logits.float() / temperature
    vocab = student.shape[-1]
    use_topk = topk is not None and topk > 0 and topk < vocab
    if use_topk:
        student_mass, student_logp, teacher_mass, teacher_logp = _topk_mass_logp(
            student, teacher, topk=int(topk), add_tail=add_tail
        )
        per_token = _divergence_from_mass_logp(
            student_mass,
            student_logp,
            teacher_mass,
            teacher_logp,
            divergence,
        )
    else:
        student_logp = F.log_softmax(student, dim=-1)
        teacher_logp = F.log_softmax(teacher, dim=-1).detach()
        per_token = _divergence_from_logp(student_logp, teacher_logp, divergence)
    return (temperature**2) * per_token.mean()


def token_greedy_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float = 0.1,
    ignore_index: int = -100,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Strongest-competitor hinge loss over causal-LM target tokens.

    ``logits[:, t]`` is aligned with ``labels[:, t + 1]``. Each sample is first
    averaged over its non-ignored target positions, then samples are averaged
    (or weighted by ``sample_weights``). The loss at a token is zero once its
    gold logit exceeds every competing logit by at least ``margin``.
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if labels.shape != logits.shape[:2]:
        raise ValueError("labels must match logits' batch and sequence dimensions")
    if logits.shape[-1] < 2:
        raise ValueError("margin loss requires a vocabulary with at least two tokens")
    if margin < 0:
        raise ValueError(f"margin must be non-negative, got {margin}")

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:].to(device=logits.device)
    valid = shift_labels.ne(ignore_index)
    if not valid.any():
        return logits.float().sum() * 0.0

    safe_labels = shift_labels.masked_fill(~valid, 0)
    target_logits = shift_logits.gather(
        dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)

    # top-2 avoids materializing a [B, T, V] one-hot mask. If gold is top-1,
    # top-2 is its strongest competitor; otherwise top-1 is the competitor.
    top_values, top_indices = shift_logits.topk(k=2, dim=-1)
    strongest_competitor = torch.where(
        top_indices[..., 0].eq(safe_labels),
        top_values[..., 1],
        top_values[..., 0],
    )
    per_token = F.relu(
        margin + strongest_competitor.float() - target_logits.float()
    )

    valid_f = valid.to(dtype=per_token.dtype)
    valid_per_sample = valid.any(dim=-1)
    per_sample = (per_token * valid_f).sum(dim=-1) / valid_f.sum(dim=-1).clamp(
        min=1
    )

    if sample_weights is None:
        weights = valid_per_sample.to(dtype=per_sample.dtype)
    else:
        if sample_weights.ndim != 1 or sample_weights.shape[0] != logits.shape[0]:
            raise ValueError("sample_weights must have shape [batch]")
        weights = sample_weights.to(device=logits.device, dtype=per_sample.dtype)
        weights = weights * valid_per_sample.to(dtype=per_sample.dtype)

    return (per_sample * weights).sum() / weights.sum().clamp(min=1e-12)


# Registry of built-in auxiliary loss functions
LOSS_REGISTRY = {
    "kl": kl_divergence,
    "l2": l2_reg,
    "mm": manifold_matching_loss,
    "mm_topk": manifold_matching_loss_topk_anchors,
    "rank": semantic_ranking_loss,
    "rank_topk": semantic_ranking_loss_topk_anchors,
    "sdpo": sdpo_distillation_loss,
    "none": none_loss,
}
