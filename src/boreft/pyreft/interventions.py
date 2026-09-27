import numbers
from typing import Optional

import torch
from pyvene import (
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
)
from pyvene.models.layers import LowRankRotateLayer
from transformers.activations import ACT2FN
import math
from .bias_network import TargetBiasNetwork
from .losses import (
    LOGVAR_MAX,
    LOGVAR_MIN,
    LOSS_REGISTRY,
    clamp_logvar,
    manifold_matching_loss,
    manifold_matching_loss_topk_anchors,
    semantic_ranking_loss,
    semantic_ranking_loss_topk_anchors,
)


# Runtime buffers that older ``intkey_*.bin`` files may still contain.
# Eval reloads ``embed_cache`` via embed_cache_path; encoder penult/attn are
# rebuilt at train time. Not learnable — ``--init-from`` skips the same names.
_RUNTIME_BUFFER_KEYS = frozenset(
    {
        "embed_cache",
        "encoder_penult",
        "encoder_attn",
        "encoder_instruction_mask",
    }
)


def _register_embed_cache(module, embed_cache):
    # Frozen target embeddings for bias-network / MM / rank. Not persisted
    # in intervenable_model weights; callers reload via embed_cache_path.
    if embed_cache is not None:
        module.register_buffer(
            "embed_cache", embed_cache.detach().float(), persistent=False
        )
    else:
        module.embed_cache = None


def _is_runtime_buffer_key(key: str) -> bool:
    return key.rsplit(".", 1)[-1] in _RUNTIME_BUFFER_KEYS


def _build_bias_network(kwargs, *, low_rank_dim: int, learnable_logvar: bool):
    # Input dim for the bias-network MLP: explicit override (e.g. LLM encoder
    # hidden size) takes precedence, otherwise the frozen embed_cache width.
    input_dim = kwargs.get("bias_network_input_dim")
    if input_dim is None:
        embed_cache = kwargs.get("embed_cache")
        if embed_cache is None:
            raise ValueError(
                "add_bias_network requires embed_cache or bias_network_input_dim"
            )
        input_dim = embed_cache.shape[1]
    return TargetBiasNetwork(
        embed_dim=int(input_dim),
        bias_dim=low_rank_dim,
        learnable_logvar=learnable_logvar,
        residual=bool(kwargs.get("bias_network_residual", False)),
    )


class _EncoderBiasMixin:
    """Shared plumbing for feeding an LLM encoder into the bias network.

    ``bias_input_source`` selects the bias-network input:
      - ``"embed_cache"`` (default): frozen sentence-transformer rows
        (``embed_cache[word_ids]``), the original behaviour.
      - ``"llm_encoder"``: a :class:`~boreft.pyreft.semantic_encoder.SemanticEncoder`
        run over cached penultimate hidden states of each word's definition text.

    The encoder is a **registered submodule** (``self.semantic_encoder``) so it
    participates in ``.train()/.eval()`` propagation, device moves, the optimizer
    parameter set, and gradient clipping. To keep checkpoints small, the
    intervention's ``state_dict`` persists only the encoder's LoRA tensors; the
    frozen copied base weights are rebuilt from the base model at load time (see
    :func:`boreft.pyreft.semantic_encoder.build_semantic_encoder`). Per-word
    penultimate inputs are cached as non-persistent buffers.
    """

    _ENCODER_ATTR = "semantic_encoder"

    def set_semantic_encoder(self, encoder) -> None:
        # Registered submodule: follows eval()/train(), device, optimizer, DDP.
        self.semantic_encoder = encoder

    def get_semantic_encoder(self):
        return getattr(self, "semantic_encoder", None)

    def set_encoder_inputs(
        self,
        penult: torch.Tensor,
        attn: torch.Tensor,
        instruction_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Cache per-word penultimate hidden states + attention masks (row = id)."""
        self.register_buffer("encoder_penult", penult.detach(), persistent=False)
        self.register_buffer("encoder_attn", attn.detach(), persistent=False)
        if instruction_mask is not None:
            self.register_buffer(
                "encoder_instruction_mask",
                instruction_mask.detach(),
                persistent=False,
            )
        else:
            self.encoder_instruction_mask = None

    def _encoder_instruction_mask_for(
        self, word_ids: torch.Tensor
    ) -> Optional[torch.Tensor]:
        mask = getattr(self, "encoder_instruction_mask", None)
        if mask is None:
            return None
        return mask[word_ids]

    def _bias_features(self, word_ids: torch.Tensor) -> torch.Tensor:
        """Return the bias-network input rows for ``word_ids``."""
        if getattr(self, "bias_input_source", "embed_cache") == "llm_encoder":
            encoder = self.get_semantic_encoder()
            if encoder is None:
                raise RuntimeError(
                    "bias_input_source='llm_encoder' requires set_semantic_encoder(...)"
                )
            penult = getattr(self, "encoder_penult", None)
            attn = getattr(self, "encoder_attn", None)
            if penult is None or attn is None:
                raise RuntimeError(
                    "bias_input_source='llm_encoder' requires set_encoder_inputs(...)"
                )
            instr = self._encoder_instruction_mask_for(word_ids)
            return encoder(penult[word_ids], attn[word_ids], instruction_mask=instr)
        return self._target_embeddings(word_ids)

    def bias_features(self, word_ids: torch.Tensor) -> torch.Tensor:
        """Public accessor for the bias-network input rows (used by materialize)."""
        return self._bias_features(word_ids)

    def set_forced_bias(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor | None = None,
    ) -> None:
        """Override per-word bias lookup with a single explicit vector.

        Used by per-instance bias learning (``/learn`` in the REPL): the bias
        network / embedding tables are bypassed and every queried word id resolves
        to ``mu`` (and optionally ``logvar``). ``mu`` is kept as-is (not detached)
        so gradients flow to the learnable vector through the normal integer-word-id
        forward path — the same path CE, SDPO scoring, and on-policy rollouts use.
        Pass a leaf ``nn.Parameter``/tensor with ``requires_grad=True`` to optimize
        it. Call :meth:`clear_forced_bias` to restore normal lookup.
        """
        self._forced_bias_mu = mu
        self._forced_bias_logvar = logvar

    def clear_forced_bias(self) -> None:
        """Remove a forced bias override set by :meth:`set_forced_bias`."""
        self._forced_bias_mu = None
        self._forced_bias_logvar = None

    def _forced_bias_active(self) -> bool:
        return getattr(self, "_forced_bias_mu", None) is not None

    def _expand_forced_mu(self, word_ids: torch.Tensor) -> torch.Tensor:
        """Broadcast the forced bias vector to one row per queried word id."""
        mu = self._forced_bias_mu
        if mu.dim() == 1:
            mu = mu.unsqueeze(0)
        return mu.expand(word_ids.shape[0], -1)

    def _expand_forced_logvar(
        self, mu_rows: torch.Tensor, fixed_logvar: float | None
    ) -> torch.Tensor:
        """Matching logvar rows for a forced bias (explicit, fixed, or prior)."""
        logvar = getattr(self, "_forced_bias_logvar", None)
        if logvar is not None:
            if logvar.dim() == 1:
                logvar = logvar.unsqueeze(0)
            return logvar.expand(mu_rows.shape[0], -1)
        if fixed_logvar is not None:
            return torch.full_like(mu_rows, fixed_logvar)
        # No explicit or fixed variance: fall back to the prior (variance=1).
        return torch.zeros_like(mu_rows)

    # ── Learnable bias table (batched per-instance learning) ──────────────────
    # Generalizes the single forced-bias vector to a trainable ``[N, low_rank]``
    # table indexed by (dummy) word id. Registered as ``nn.Parameter``(s) so the
    # standard optimizer over ``reft_model.parameters()`` picks them up while every
    # other component stays frozen — this lets the main ``ReftTrainer`` code path
    # fit biases with any recipe (CE / VAE / SDPO). Warm-start ``mu_init`` from the
    # bias network's ``mu_pred``.

    def set_learnable_bias_table(
        self,
        mu_init: torch.Tensor,
        logvar_init: torch.Tensor | None = None,
    ) -> None:
        """Register a trainable per-row bias table (row ``i`` ↔ word id ``i``).

        ``mu_init`` is ``[N, low_rank]``. When ``logvar_init`` (same shape) is
        given, a trainable log-variance table is also registered (VAE checkpoints
        with learnable variance). Call :meth:`clear_learnable_bias_table` to remove.
        """
        import torch.nn as nn

        self.clear_learnable_bias_table()
        self._learn_bias_mu = nn.Parameter(mu_init.detach().clone().float())
        if logvar_init is not None:
            self._learn_bias_logvar = nn.Parameter(
                logvar_init.detach().clone().float()
            )
        self._learn_bias_active_flag = True

    def clear_learnable_bias_table(self) -> None:
        """Remove a learnable bias table set by :meth:`set_learnable_bias_table`."""
        for name in ("_learn_bias_mu", "_learn_bias_logvar"):
            if name in getattr(self, "_parameters", {}):
                del self._parameters[name]
            elif hasattr(self, name):
                delattr(self, name)
        self._learn_bias_active_flag = False

    def _learnable_bias_active(self) -> bool:
        return bool(getattr(self, "_learn_bias_active_flag", False))

    def _learn_mu_rows(self, word_ids: torch.Tensor) -> torch.Tensor:
        """Gather trainable mean-bias rows for the queried word ids."""
        return self._learn_bias_mu[word_ids]

    def _learn_logvar_rows(
        self, mu_rows: torch.Tensor, word_ids: torch.Tensor, fixed_logvar: float | None
    ) -> torch.Tensor:
        """Matching logvar rows for the learnable table (learned, fixed, or prior)."""
        logvar = getattr(self, "_learn_bias_logvar", None)
        if logvar is not None:
            return logvar[word_ids]
        if fixed_logvar is not None:
            return torch.full_like(mu_rows, fixed_logvar)
        return torch.zeros_like(mu_rows)

    def _shared_bias_store(self) -> dict | None:
        """Per-step bias-network cache (word_id -> {mu, logvar?}), or ``None``."""
        return getattr(self, "_shared_bias", None)

    def _populate_shared_bias_rows(
        self, word_ids: torch.Tensor, *, need_logvar: bool
    ) -> None:
        """Compute and cache bias-network rows for any missing word ids in the store."""
        store = self._shared_bias_store()
        if store is None:
            return
        ids = word_ids.tolist()
        missing = [w for w in ids if w not in store]
        if not missing:
            return
        miss_t = torch.tensor(missing, dtype=torch.long, device=word_ids.device)
        if need_logvar:
            mu, logvar = self.bias_network(self._bias_features(miss_t))
            if logvar is None:
                logvar = torch.full_like(mu, self.fixed_logvar)
            for w, m, lv in zip(missing, mu, logvar):
                store[w] = {"mu": m, "logvar": lv}
        else:
            mu = self.bias_network.mu(self._bias_features(miss_t))
            for w, m in zip(missing, mu):
                store[w] = {"mu": m}

    def _bias_network_mu(self, word_ids: torch.Tensor) -> torch.Tensor:
        """Bias-network mean with optional per-step cache (CE/SDPO sharing)."""
        store = self._shared_bias_store()
        if store is None:
            return self.bias_network.mu(self._bias_features(word_ids))
        need_logvar = bool(getattr(self.bias_network, "learnable_logvar", False))
        self._populate_shared_bias_rows(word_ids, need_logvar=need_logvar)
        return torch.stack([store[w]["mu"] for w in word_ids.tolist()], dim=0)

    def _bias_network_mu_logvar(
        self, word_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bias-network (mu, logvar) with optional per-step cache (CE/SDPO sharing)."""
        store = self._shared_bias_store()
        if store is None:
            mu, logvar = self.bias_network(self._bias_features(word_ids))
            if logvar is None:
                logvar = torch.full_like(mu, self.fixed_logvar)
            return mu, logvar
        self._populate_shared_bias_rows(word_ids, need_logvar=True)
        ids = word_ids.tolist()
        mu = torch.stack([store[w]["mu"] for w in ids], dim=0)
        logvar = torch.stack([store[w]["logvar"] for w in ids], dim=0)
        return mu, logvar

    @staticmethod
    def _is_encoder_frozen_key(key: str) -> bool:
        """True for encoder keys that are rebuilt from the base (not persisted)."""
        return "semantic_encoder." in key and ".lora_" not in key

    def state_dict(self, *args, **kwargs):
        """Drop the encoder's frozen copied base weights from the saved state.

        Only the LoRA tensors under ``semantic_encoder.`` are kept; the frozen
        copied decoder block + norm are reconstructed from the base model at load
        time. Works for both the direct call (pyvene ``save_intervention``) and
        the recursive call from a parent module.
        """
        sd = super().state_dict(*args, **kwargs)
        if self.get_semantic_encoder() is None:
            return sd
        for key in [k for k in sd if self._is_encoder_frozen_key(k)]:
            del sd[key]
        return sd

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load intervention weights, skipping runtime buffers and frozen encoder keys.

        The encoder's frozen copied weights are not persisted (see
        :meth:`state_dict`); they are already present from reconstruction, so we
        allow them to be "missing" while still enforcing strictness for every
        other key. ``embed_cache`` and encoder penult/attn buffers are runtime
        data (older checkpoints may still serialize ``embed_cache``).
        """
        incompatible = super().load_state_dict(state_dict, strict=False, assign=assign)
        disallowed_missing = [
            k
            for k in incompatible.missing_keys
            if not self._is_encoder_frozen_key(k) and not _is_runtime_buffer_key(k)
        ]
        disallowed_unexpected = [
            k
            for k in incompatible.unexpected_keys
            if not _is_runtime_buffer_key(k)
        ]
        if strict and (disallowed_missing or disallowed_unexpected):
            msgs = []
            if disallowed_unexpected:
                msgs.append(f"Unexpected key(s): {disallowed_unexpected}")
            if disallowed_missing:
                msgs.append(f"Missing key(s): {disallowed_missing}")
            raise RuntimeError(
                "Error(s) in loading state_dict for "
                f"{self.__class__.__name__}: {'; '.join(msgs)}"
            )
        return incompatible


class DistributionalWordIntervention(
    _EncoderBiasMixin,
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
):
    """
    LoReFT intervention with optional per-word distributional bias.

    use_word_bias=True  (default): h + R^T(Wh + b_w - Rh)
        Each word w has a learned Gaussian N(mu_w, sigma_w^2) over b.
        Subspaces must carry integer word IDs: [[word_id], ...].

        Bias source (mutually exclusive when use_word_bias=True):
        - Default: per-word ``word_mu`` / ``word_logvar`` embedding tables.
        - ``add_bias_network=True``: shared MLP over bias-network inputs — either
          frozen sentence-transformer ``embed_cache`` rows (default) or, with
          ``bias_input_source='llm_encoder'``, pooled features from a copied final
          decoder block + LoRA over each word's definition text. MM/rank auxiliary
          losses are disabled in that mode. After training, ``bias_tables.pt`` can
          be exported for O(1) eval lookup.

    use_word_bias=False: h + R^T(Wh - Rh)
        Standard sourceless LoReFT; subspaces are not required.

    variance: "learnable" (default) or a positive float scalar.
        "learnable"  — per-word logvar is a trainable nn.Embedding, initialised to 0 (variance=1).
        float scalar — variance is fixed to that value for all words (no word_logvar table).
                       e.g. variance=0.01 sets std=0.1 at every word, every step.
                       Allowed range: (e^-14, e^10] so logvar stays in [LOGVAR_MIN, LOGVAR_MAX].
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"])
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"], bias=False
        ).to(kwargs.get("dtype", torch.bfloat16))
        self.dropout = torch.nn.Dropout(kwargs.get("dropout", 0.0))
        self.dropout_on_b = torch.nn.Dropout(kwargs.get("dropout_on_b", 0.0))
        self.act_fn = (
            ACT2FN["linear"]
            if kwargs.get("act_fn") is None
            else ACT2FN[kwargs["act_fn"]]
        )

        self.use_word_bias = kwargs.get("use_word_bias", True)

        # Optional per-step shared reparameterized bias (word_id -> sampled b row,
        # *with* its grad path). When the trainer activates this (sets a fresh dict
        # each step), the CE forward records one sampled b per word and the SDPO
        # scoring forward reuses the same tensor, so both objectives backpropagate
        # through a single reparameterization node. ``None`` (default) restores
        # independent sampling on every forward.
        self._shared_b: dict | None = None
        # Optional per-step bias-network cache (word_id -> {mu, logvar?}). When the
        # trainer activates this alongside ``_shared_b``, the CE forward records
        # encoder + bias-network rows once per word and the SDPO scoring forward
        # reuses the same tensors (same grad path through encoder LoRA / MLP).
        self._shared_bias: dict | None = None

        # Per-loss coefficients — 0.0 disables that term.
        self.kl_beta = float(kwargs.get("kl_beta", 0.0))
        self.kl_prior_var = float(kwargs.get("kl_prior_var", 1.0))  # prior variance τ²
        if self.kl_prior_var <= 0:
            raise ValueError(f"kl_prior_var must be positive, got {self.kl_prior_var}")
        # Free-bits floor (nats per latent dim); 0 disables. See losses.kl_divergence.
        self.free_bits_lambda = float(kwargs.get("vae_free_bits_lambda", 0.0))
        if self.free_bits_lambda < 0:
            raise ValueError(
                f"vae_free_bits_lambda must be non-negative, got {self.free_bits_lambda}"
            )
        self.lambda_l2 = float(kwargs.get("lambda_l2", 0.0))

        # variance: "learnable" or a fixed positive float
        variance = kwargs.get("variance", "learnable")
        if isinstance(variance, str):
            if variance != "learnable":
                raise ValueError(
                    f"variance must be 'learnable' or a positive float, got {variance!r}"
                )
            self.fixed_logvar = None
        else:
            logvar_val = math.log(float(variance))
            if not (LOGVAR_MIN <= logvar_val <= LOGVAR_MAX):
                raise ValueError(
                    f"variance={variance} gives logvar={logvar_val:.2f} which is outside the "
                    f"stable range [{LOGVAR_MIN}, {LOGVAR_MAX}]."
                )
            self.fixed_logvar = logvar_val

        self.add_bias_network = bool(kwargs.get("add_bias_network", False))
        self.bias_network_residual = bool(kwargs.get("bias_network_residual", False))
        self.bias_input_source = kwargs.get("bias_input_source", "embed_cache")
        self.semantic_encoder = None
        embed_cache = kwargs.get("embed_cache", None)
        _register_embed_cache(self, embed_cache)

        if self.use_word_bias:
            num_words = kwargs.get("num_words", 1)
            sigma_init = kwargs.get("sigma_b", 0.1)
            low_rank_dim = kwargs["low_rank_dimension"]

            if self.add_bias_network:
                self.bias_network = _build_bias_network(
                    kwargs,
                    low_rank_dim=low_rank_dim,
                    learnable_logvar=self.fixed_logvar is None,
                )
                self.word_mu = None
                self.word_logvar = None
            else:
                self.bias_network = None
                self.word_mu = torch.nn.Embedding(num_words, low_rank_dim)
                torch.nn.init.normal_(self.word_mu.weight, mean=0.0, std=sigma_init)

                if self.fixed_logvar is None:
                    # learnable: one logvar per word per dimension, initialised to 0 (variance=1=prior)
                    self.word_logvar = torch.nn.Embedding(num_words, low_rank_dim)
                    torch.nn.init.zeros_(self.word_logvar.weight)
        else:
            self.bias_network = None

        # Manifold matching: ref[i] = precomputed emb(target) for word id i; λ_mm scales mm_loss.
        self.lambda_mm = float(kwargs.get("lambda_mm", 0.0))
        self.mm_exclude_diagonal = bool(kwargs.get("mm_exclude_diagonal", True))
        # MM over full vocab: top-k ref neighbours + random t batch anchors (0 / 0 = batch-only MM).
        self.mm_topk_neighbors = int(kwargs.get("mm_topk_neighbors", 0))
        self.mm_anchor_samples = int(kwargs.get("mm_anchor_samples", 0))

        # Semantic ranking loss (PRO-style batch-wise ranking).
        self.lambda_rank = float(kwargs.get("lambda_rank", 0.0))
        self.rank_temperature = float(kwargs.get("rank_temperature", 0.1))
        self.rank_min_ref_gap = float(kwargs.get("rank_min_ref_gap", 0.0))
        # Full-vocab top-k for ranking: same semantics as mm_topk_neighbors/mm_anchor_samples.
        self.rank_topk_neighbors = int(kwargs.get("rank_topk_neighbors", 0))
        self.rank_anchor_samples = int(kwargs.get("rank_anchor_samples", 0))

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, exp(logvar)).

        Args:
            mu (torch.Tensor): Mean of the distribution.
            logvar (torch.Tensor): Log-variance of the distribution.

        Returns:
            torch.Tensor: Sampled tensor.
        """
        logvar = clamp_logvar(logvar)

        # Calculate standard deviation
        std = torch.exp(0.5 * logvar)

        # Removed the below additional clamping
        # Clamp std to avoid extreme values
        # std = torch.clamp(std, min=1e-4, max=1.0)

        # Sample from standard normal distribution
        eps = torch.randn_like(std)

        # Return the sampled value with reparameterization
        return mu + eps * std

    def _bias_sample_rows(self, mu, logvar, word_ids) -> torch.Tensor:
        """Reparameterized sample b ~ N(mu, exp(logvar)); one row per word id.

        When a per-step shared-b store is active (``self._shared_b`` is a dict, set
        by the trainer), the sampled row for each word id is recorded on first use
        and the *same tensor* — retaining its grad path — is returned on subsequent
        lookups. The companion ``_shared_bias`` store does the same for bias-network
        (encoder + MLP) rows so CE and SDPO do not recompute them. When the store
        is ``None`` (default), each call samples independently (original behaviour).
        """
        store = self._shared_b
        if store is None:
            return self.dropout_on_b(self.reparameterize(mu, logvar))
        ids = word_ids.tolist()
        missing = [i for i, w in enumerate(ids) if w not in store]
        if missing:
            # Cache the *post-dropout* bias so CE and SDPO share the identical b
            # (same noise and same dropout mask), not just the pre-dropout sample.
            sampled = self.dropout_on_b(self.reparameterize(mu, logvar))
            for i in missing:
                store[ids[i]] = sampled[i]
        return torch.stack([store[w] for w in ids], dim=0)

    def _target_embeddings(self, word_ids: torch.Tensor) -> torch.Tensor:
        if self.embed_cache is None:
            raise RuntimeError("embed_cache is required for bias-network lookup")
        return self.embed_cache[word_ids]

    def _lookup_materialized_mu(self, word_ids: torch.Tensor) -> torch.Tensor | None:
        tables = getattr(self, "materialized_mu", None)
        if tables is None:
            return None
        return tables[word_ids]

    def _lookup_materialized_logvar(
        self, word_ids: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor | None:
        tables = getattr(self, "materialized_logvar", None)
        if tables is not None:
            return tables[word_ids]
        if getattr(self, "fixed_logvar", None) is not None:
            return torch.full_like(mu, self.fixed_logvar)
        return None

    def get_bias_mu(self, word_ids: torch.Tensor) -> torch.Tensor:
        """Per-word mean bias vectors [len(word_ids), low_rank]."""
        if self._learnable_bias_active():
            return self._learn_mu_rows(word_ids)
        if self._forced_bias_active():
            return self._expand_forced_mu(word_ids)
        materialized = self._lookup_materialized_mu(word_ids)
        if materialized is not None:
            return materialized
        if self.bias_network is not None:
            return self._bias_network_mu(word_ids)
        return self.word_mu(word_ids)

    def get_bias_mu_logvar(
        self, word_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-word (mu, logvar) with one bias-network forward when applicable."""
        if self._learnable_bias_active():
            mu = self._learn_mu_rows(word_ids)
            return mu, self._learn_logvar_rows(mu, word_ids, self.fixed_logvar)
        if self._forced_bias_active():
            mu = self._expand_forced_mu(word_ids)
            return mu, self._expand_forced_logvar(mu, self.fixed_logvar)
        materialized_mu = self._lookup_materialized_mu(word_ids)
        if materialized_mu is not None:
            logvar = self._lookup_materialized_logvar(word_ids, materialized_mu)
            if logvar is None:
                raise RuntimeError(
                    "materialized_mu is set but logvar is missing "
                    "(expected materialized_logvar or fixed_logvar)"
                )
            return materialized_mu, logvar
        if self.bias_network is not None:
            return self._bias_network_mu_logvar(word_ids)
        mu = self.word_mu(word_ids)
        if self.fixed_logvar is not None:
            logvar = torch.full_like(mu, self.fixed_logvar)
        else:
            logvar = self.word_logvar(word_ids)
        return mu, logvar

    def auxiliary_loss(self) -> torch.Tensor:
        """Sum of enabled auxiliary losses, each scaled by its own coefficient.

        kl_beta   > 0  →  kl_beta   * KL(N(mu,sigma) || N(0,kl_prior_var))
        lambda_l2 > 0  →  lambda_l2 * ||mu||²
        lambda_mm > 0  →  lambda_mm * manifold_matching_loss(mu, ref)
        Set a coefficient to 0.0 to disable that term.

        When ``free_bits_lambda`` > 0, the KL term applies a free-bits floor: each
        latent dimension's batch-averaged KL is clamped to be at least
        ``free_bits_lambda`` nats before summing, so KL below the floor contributes
        no gradient (mitigates posterior collapse). See ``losses.kl_divergence``.
        """
        device = self.rotate_layer.weight.device
        total = torch.tensor(0.0, device=device)

        last_mu = getattr(self, "last_mu", None)
        last_logvar = getattr(self, "last_logvar", None)
        last_wid = getattr(self, "last_word_ids", None)

        if self.use_word_bias and last_mu is not None and last_logvar is not None:
            if self.kl_beta > 0.0:
                kl_fn = LOSS_REGISTRY["kl"]
                total = total + self.kl_beta * kl_fn(
                    last_mu,
                    last_logvar,
                    prior_var=self.kl_prior_var,
                    free_bits=getattr(self, "free_bits_lambda", 0.0),
                )

            if self.lambda_l2 > 0.0:
                l2_fn = LOSS_REGISTRY["l2"]
                total = total + self.lambda_l2 * l2_fn(last_mu, last_logvar)

        if (
            # MM loss requires a static z table; skip when bias comes from f(e).
            not self.add_bias_network
            and self.lambda_mm > 0.0
            and self.embed_cache is not None
            and last_wid is not None
            and last_mu is not None
        ):
            if self.mm_topk_neighbors > 0 and self.use_word_bias:
                mm_loss = manifold_matching_loss_topk_anchors(
                    self.embed_cache,
                    self.word_mu.weight,
                    last_wid,
                    k=self.mm_topk_neighbors,
                    t=self.mm_anchor_samples,
                    exclude_diagonal=self.mm_exclude_diagonal,
                )
            else:
                ref_b = self.embed_cache[last_wid]
                mm_loss = manifold_matching_loss(
                    last_mu,
                    ref_b,
                    exclude_diagonal=self.mm_exclude_diagonal,
                )
            total = total + self.lambda_mm * mm_loss

        if (
            not self.add_bias_network
            and self.lambda_rank > 0.0
            and self.embed_cache is not None
            and last_wid is not None
            and last_mu is not None
        ):
            if self.rank_topk_neighbors > 0 and self.use_word_bias:
                rank_loss = semantic_ranking_loss_topk_anchors(
                    self.embed_cache,
                    self.word_mu.weight,
                    last_wid,
                    k=self.rank_topk_neighbors,
                    t=self.rank_anchor_samples,
                    temperature=self.rank_temperature,
                    exclude_diagonal=self.mm_exclude_diagonal,
                    min_ref_gap=self.rank_min_ref_gap,
                )
            else:
                ref_b = self.embed_cache[last_wid]
                rank_loss = semantic_ranking_loss(
                    last_mu,
                    ref_b,
                    temperature=self.rank_temperature,
                    exclude_diagonal=self.mm_exclude_diagonal,
                    min_ref_gap=self.rank_min_ref_gap,
                )
            total = total + self.lambda_rank * rank_loss

        return total

    def forward(self, base, source=None, subspaces=None):
        rotated_base = self.rotate_layer(base)
        wh = self.act_fn(self.learned_source(base))

        if self.use_word_bias:
            first_val = (
                subspaces[0][0]
                if isinstance(subspaces[0], (list, tuple))
                else subspaces[0]
            )
            if isinstance(first_val, numbers.Integral):
                # Word ID lookup
                word_ids = torch.tensor(
                    [s[0] if isinstance(s, (list, tuple)) else s for s in subspaces],
                    dtype=torch.long,
                    device=wh.device,
                )
                self.last_word_ids = word_ids
                mu, logvar = self.get_bias_mu_logvar(word_ids)
                logvar = clamp_logvar(logvar)
                if self.training:
                    # _bias_sample_rows applies dropout_on_b (and caches the
                    # post-dropout b when sharing), so it is not re-applied here.
                    b = self._bias_sample_rows(mu, logvar, word_ids).to(
                        dtype=wh.dtype
                    ).unsqueeze(1)
                else:
                    b = self.dropout_on_b(
                        mu.to(dtype=wh.dtype).unsqueeze(1)
                    )
                self.last_mu = mu
                self.last_logvar = logvar
            else:
                # Raw b vector (e.g. for interpolation) — no word ids / no KL-mm state
                self.last_word_ids = None
                self.last_mu = None
                self.last_logvar = None
                b_raw = torch.tensor(subspaces, dtype=wh.dtype, device=wh.device)
                if b_raw.dim() == 2:
                    b_raw = b_raw.unsqueeze(1)
                elif b_raw.dim() == 1:
                    b_raw = b_raw.unsqueeze(0).unsqueeze(0)
                b = b_raw

            wh = wh + b

        delta = wh - rotated_base
        output = base + torch.matmul(delta, self.rotate_layer.weight.T)
        return self.dropout(output.to(base.dtype))


class LoreftPerWordBiasIntervention(
    _EncoderBiasMixin,
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
):
    """
    LoReFT with a learned per-word bias vector b_w (input-independent).
    h_out = h + R^T( Wh + b_w - Rh )
    where Wh = act_fn(W h) and Rh is the shared low-rank rotation of h.

    Each word w has its own bias vector b_w stored in a plain nn.Embedding
    table of shape [num_words, low_rank_dimension]. Unlike DistributionalWord
    Intervention there is no variance / sampling — b_w is always used
    deterministically.

    Subspaces behaviour:
    - Integer word ids  → looks up b_w from the embedding table.
    - Raw float vectors → uses them directly as b (e.g. for interpolation).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.num_words = kwargs["num_words"]

        low_rank = kwargs["low_rank_dimension"]
        dtype = kwargs.get("dtype", torch.bfloat16)
        sigma_init = kwargs.get("sigma_b", 1.0)

        # Shared orthogonal rotation R
        rotate_layer = LowRankRotateLayer(self.embed_dim, low_rank)
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)

        # Shared W
        self.learned_source = torch.nn.Linear(self.embed_dim, low_rank, bias=False).to(
            dtype
        )

        self.add_bias_network = bool(kwargs.get("add_bias_network", False))
        self.bias_network_residual = bool(kwargs.get("bias_network_residual", False))
        self.bias_input_source = kwargs.get("bias_input_source", "embed_cache")
        self.semantic_encoder = None
        embed_cache = kwargs.get("embed_cache", None)
        _register_embed_cache(self, embed_cache)

        if self.add_bias_network:
            self.bias_network = _build_bias_network(
                kwargs,
                low_rank_dim=low_rank,
                learnable_logvar=False,
            )
            self.word_bias = None
        else:
            self.bias_network = None
            # Per-word bias vectors b_w  [num_words, low_rank]
            self.word_bias = torch.nn.Embedding(self.num_words, low_rank)
            torch.nn.init.normal_(self.word_bias.weight, mean=0.0, std=sigma_init)

        self.dropout = torch.nn.Dropout(kwargs.get("dropout", 0.0))
        self.dropout_on_b = torch.nn.Dropout(kwargs.get("dropout_on_b", 0.0))
        self.act_fn = (
            ACT2FN["linear"]
            if kwargs.get("act_fn") is None
            else ACT2FN[kwargs["act_fn"]]
        )

        # lambda_mm > 0 enables manifold-matching loss; 0.0 disables it.
        self.lambda_mm = float(kwargs.get("lambda_mm", 0.0))
        self.mm_exclude_diagonal = bool(kwargs.get("mm_exclude_diagonal", True))
        self.mm_topk_neighbors = int(kwargs.get("mm_topk_neighbors", 0))
        self.mm_anchor_samples = int(kwargs.get("mm_anchor_samples", 0))

        # Semantic ranking loss (PRO-style batch-wise ranking).
        self.lambda_rank = float(kwargs.get("lambda_rank", 0.0))
        self.rank_temperature = float(kwargs.get("rank_temperature", 0.1))
        self.rank_min_ref_gap = float(kwargs.get("rank_min_ref_gap", 0.0))
        # Full-vocab top-k for ranking: same semantics as mm_topk_neighbors/mm_anchor_samples.
        self.rank_topk_neighbors = int(kwargs.get("rank_topk_neighbors", 0))
        self.rank_anchor_samples = int(kwargs.get("rank_anchor_samples", 0))

        # Per-step bias-network cache for CE/SDPO sharing (see ``_shared_bias`` on
        # :class:`DistributionalWordIntervention``).
        self._shared_bias: dict | None = None

    def _target_embeddings(self, word_ids: torch.Tensor) -> torch.Tensor:
        if self.embed_cache is None:
            raise RuntimeError("embed_cache is required for bias-network lookup")
        return self.embed_cache[word_ids]

    def get_bias_mu(self, word_ids: torch.Tensor) -> torch.Tensor:
        """Per-word bias vectors [len(word_ids), low_rank]."""
        if self._learnable_bias_active():
            return self._learn_mu_rows(word_ids)
        if self._forced_bias_active():
            return self._expand_forced_mu(word_ids)
        tables = getattr(self, "materialized_mu", None)
        if tables is not None:
            return tables[word_ids]
        if self.bias_network is not None:
            return self._bias_network_mu(word_ids)
        return self.word_bias(word_ids)

    def _bias_param_device(self):
        if self.bias_network is not None:
            return next(self.bias_network.parameters()).device
        return self.word_bias.weight.device

    def auxiliary_loss(self) -> torch.Tensor:
        """λ_mm · manifold_matching_loss + λ_rank · semantic_ranking_loss."""
        device = self._bias_param_device()
        total = torch.tensor(0.0, device=device)
        last_z = getattr(self, "last_z", None)
        last_wid = getattr(self, "last_word_ids", None)

        if (
            # MM loss requires a static z table; skip when bias comes from f(e).
            not self.add_bias_network
            and self.lambda_mm > 0.0
            and self.embed_cache is not None
            and last_z is not None
            and last_wid is not None
        ):
            if self.mm_topk_neighbors > 0:
                mm_loss = manifold_matching_loss_topk_anchors(
                    self.embed_cache,
                    self.word_bias.weight,
                    last_wid,
                    k=self.mm_topk_neighbors,
                    t=self.mm_anchor_samples,
                    exclude_diagonal=self.mm_exclude_diagonal,
                )
            else:
                ref_b = self.embed_cache[last_wid]
                mm_loss = manifold_matching_loss(
                    last_z,
                    ref_b,
                    exclude_diagonal=self.mm_exclude_diagonal,
                )
            total = total + self.lambda_mm * mm_loss

        if (
            not self.add_bias_network
            and self.lambda_rank > 0.0
            and self.embed_cache is not None
            and last_z is not None
            and last_wid is not None
        ):
            if self.rank_topk_neighbors > 0:
                rank_loss = semantic_ranking_loss_topk_anchors(
                    self.embed_cache,
                    self.word_bias.weight,
                    last_wid,
                    k=self.rank_topk_neighbors,
                    t=self.rank_anchor_samples,
                    temperature=self.rank_temperature,
                    exclude_diagonal=self.mm_exclude_diagonal,
                    min_ref_gap=self.rank_min_ref_gap,
                )
            else:
                ref_b = self.embed_cache[last_wid]
                rank_loss = semantic_ranking_loss(
                    last_z,
                    ref_b,
                    temperature=self.rank_temperature,
                    exclude_diagonal=self.mm_exclude_diagonal,
                    min_ref_gap=self.rank_min_ref_gap,
                )
            total = total + self.lambda_rank * rank_loss

        return total

    def forward(self, base, source=None, subspaces=None):
        rotated_base = self.rotate_layer(base)  # (batch, seq, low_rank)
        wh = self.act_fn(self.learned_source(base))  # (batch, seq, low_rank)

        if subspaces is None:
            self.last_word_ids = None  # Needed for auxiliary loss
            self.last_z = None
            b_terms = torch.zeros_like(wh)
        else:
            first_val = (
                subspaces[0][0]
                if isinstance(subspaces[0], (list, tuple))
                else subspaces[0]
            )

            if isinstance(first_val, numbers.Integral):
                word_ids = torch.tensor(
                    [s[0] if isinstance(s, (list, tuple)) else s for s in subspaces],
                    dtype=torch.long,
                    device=wh.device,
                )
                self.last_word_ids = word_ids
                # b_w: [batch, low_rank]; dropout before expand so forward and aux share the same b
                b = self.dropout_on_b(
                    self.get_bias_mu(word_ids).to(dtype=wh.dtype)
                )
                self.last_z = b
                b_terms = b.unsqueeze(1).expand(
                    -1, wh.shape[1], -1
                )  # [batch, seq, low_rank]
            else:
                self.last_word_ids = None
                self.last_z = None
                b_raw = torch.tensor(subspaces, dtype=wh.dtype, device=wh.device)
                if b_raw.dim() == 2:
                    b_raw = b_raw.unsqueeze(1)
                elif b_raw.dim() == 1:
                    b_raw = b_raw.unsqueeze(0).unsqueeze(0)

                if b_raw.dim() == 3 and b_raw.shape[1] == 1 and wh.shape[1] != 1:
                    b_raw = b_raw.expand(-1, wh.shape[1], -1)
                b_terms = b_raw

        delta = wh + b_terms - rotated_base
        output = base + torch.matmul(delta, self.rotate_layer.weight.T)
        return self.dropout(output.to(base.dtype))
