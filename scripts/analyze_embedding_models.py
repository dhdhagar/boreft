#!/usr/bin/env python
"""Compare candidate molopt embedding models on SMILES ↔ description alignment.

The molopt task assumes one embedding space holds both molecules and chemistry
prose. That assumption is load-bearing — ``embed_sim``, the MM/rank losses, and
the ``--use-definition-embeds`` target all rest on it — so this script measures
it across several candidate encoders instead of taking any model card's word for
it. See ``notes/embedding_models.md`` for how to read the output.

Models (``--models``, default all available):

``chemate``
    ``SchwallerGroup/CheMatE-v0``: a chemistry ModernBERT, mean-pooled, via
    sentence-transformers. Was molopt's encoder until this comparison retired it.
``molt5``
    ``laituan245/molt5-base`` encoder, mean-pooled over non-pad tokens. Its
    pretraining saw both SMILES and text but with no cross-modal objective, so
    it is the informative negative control: any alignment here is incidental.
``moleculestm``
    ``chao1224/MoleculeSTM``, the only genuinely cross-modal candidate — a GIN
    molecule tower and a SciBERT text tower trained contrastively against each
    other, each projected to a shared 256-d space. Dual-tower, so molecule-side
    prompt decoration does not apply (see ``VARIANTS``).
``qwen3``
    ``Qwen/Qwen3-Embedding-0.6B``, last token of the last hidden state, which is
    this model's native pooling. General-purpose text encoder with no chemistry
    training, and — on the strength of this comparison — what both tasks now use.
``qwen3-st``
    the same weights through sentence-transformers. Its pooling config is also
    last-token, so this exists purely to confirm the hand-rolled ``qwen3`` path
    reproduces the library one. Off by default.

Two numbers matter, and neither is the raw matched cosine:

``gap`` (matched − random)
    these encoders squash all drug-like input into a narrow high-cosine band, so
    a matched cosine of 0.85 can still sit *below* the average unrelated pair.
    Only the gap against the mismatched control is interpretable.
``retrieval``
    rank every molecule by cosine to one description and see where the true one
    lands (Recall@k, MRR, chance = 1/N). This is the decision-relevant number: at
    chance, the space cannot tell molecules apart by their descriptions no matter
    how high the matched cosine looks.

The last section checks each metric against structure directly: embedding cosine
vs. Morgan/Tanimoto over random molecule pairs, for both the ``embedding_prompt``
text that ``embed_sim`` scores and the ``embedding_prompt_defn`` text that
``--use-definition-embeds`` trains on. Weak correlation is why the eval suite
reports TFS alongside ``embed_sim`` rather than trusting either alone.

    python scripts/analyze_embedding_models.py --n 1000              # CPU-ok subset
    python scripts/analyze_embedding_models.py --n 3000 --device cuda # full run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from boreft.bo.plotting import save_plot  # noqa: E402
from boreft.chem import tanimoto_similarity  # noqa: E402
from boreft.search_wandb import add_wandb_cli, maybe_log_named_analysis  # noqa: E402
from boreft.task_config import (  # noqa: E402
    definition_embedding_template,
    definition_embedding_text,
)
from boreft.text_similarity import (  # noqa: E402
    default_definitions_path,
    embedding_prompt_template,
)

TASK = "molopt"
RECALL_KS = (1, 5, 10)

# snapshot_download keeps the repo's own "demo/" prefix, so the checkpoints land
# one level deeper than the HF repo path suggests.
MOLECULESTM_DEFAULT_DIR = os.path.join(
    "data", "molopt", "raw", "MoleculeSTM", "demo", "demo_checkpoints_Graph"
)
MOLECULESTM_DOWNLOAD_HINT = (
    "huggingface-cli download chao1224/MoleculeSTM "
    "--include 'demo/demo_checkpoints_Graph/*' "
    "--local-dir data/molopt/raw/MoleculeSTM"
)
# Buffers that newer transformers derives instead of storing (see load_into).
_STALE_BUFFER_KEYS = ("embeddings.position_ids", "embeddings.token_type_ids")

# Stand-in for the {text} slot when embedding a definition with the template but
# without the molecule (see the DEFN_NO_SMILES variant). Any constant works: the
# point is that it is identical for every row and so cannot carry row-specific
# signal. "X" is the usual notation for an unspecified group.
SMILES_PLACEHOLDER = "X"


# ─────────────────────────────────────────────────────────────────────────────
# Text treatments
#
# A variant is a pair of treatments, one per side of the comparison. Keeping them
# named and separate is what makes the leakage control below legible: the
# definition-side template contains {text}, i.e. the SMILES we are trying to
# retrieve, so a high score with DEFN_PROMPTED means little on its own.
# ─────────────────────────────────────────────────────────────────────────────

MOL_BARE = "smiles"
MOL_PROMPTED = "smiles_prompted"
DEFN_BARE = "defn"
DEFN_PROMPTED = "defn_prompted"
DEFN_NO_SMILES = "defn_prompted_no_smiles"


def molecule_text(smiles: str, treatment: str) -> str:
    if treatment == MOL_BARE:
        return smiles.strip()
    if treatment == MOL_PROMPTED:
        return embedding_prompt_template(TASK).format(text=smiles.strip())
    raise ValueError(f"unknown molecule treatment {treatment!r}")


def definition_text(smiles: str, definition: str, treatment: str) -> str:
    if treatment == DEFN_BARE:
        return definition.strip()
    if treatment == DEFN_PROMPTED:
        return definition_embedding_text(TASK, smiles, definition)
    if treatment == DEFN_NO_SMILES:
        return definition_embedding_text(TASK, SMILES_PLACEHOLDER, definition)
    raise ValueError(f"unknown definition treatment {treatment!r}")


@dataclass(frozen=True)
class Variant:
    mol: str
    defn: str
    question: str

    @property
    def name(self) -> str:
        return variant_name(self.mol, self.defn)


def variant_name(mol_treatment: str, defn_treatment: str) -> str:
    """Row label naming both sides, so a row always says what it encoded."""
    return f"{mol_treatment}__{defn_treatment}"


VARIANTS: tuple[Variant, ...] = (
    Variant(
        MOL_BARE,
        DEFN_BARE,
        "Bare SMILES vs. bare description — each model card's own use case.",
    ),
    Variant(
        MOL_PROMPTED,
        DEFN_BARE,
        "What embed_sim really encodes on the molecule side, against an "
        "undecorated definition. The honest cross-modal number.",
    ),
    Variant(
        MOL_PROMPTED,
        DEFN_PROMPTED,
        "Eval-time target text vs. the --use-definition-embeds training-target "
        "text. Low here means training pulls targets where eval does not score. "
        "Inflated: the definition template embeds the SMILES itself.",
    ),
    Variant(
        MOL_PROMPTED,
        DEFN_NO_SMILES,
        "The same definition template with the molecule slot replaced by a "
        "constant, which removes the leakage above. The drop from the previous "
        "row is how much of it was string overlap.",
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Encoders
#
# Each encoder exposes ``encode(texts) -> [n, d]`` L2-normalized rows, plus a
# ``dual_tower`` flag. Dual-tower models route molecules and text through
# different networks and take a graph rather than a string on the molecule side,
# so prompt decoration is meaningless there and those variants are skipped
# rather than silently reported as identical numbers.
# ─────────────────────────────────────────────────────────────────────────────


def _l2_normalize(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(norms, 1e-12)


class Encoder:
    """Base class; subclasses implement ``_encode_text`` (and maybe ``_encode_molecules``)."""

    key: str = ""
    label: str = ""
    dual_tower: bool = False

    def encode_text(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        return _l2_normalize(self._encode_text(list(texts), batch_size))

    def encode_molecules(
        self, decorated: Sequence[str], raw_smiles: Sequence[str], batch_size: int
    ) -> np.ndarray:
        """Molecule-side embeddings.

        Single-tower models embed ``decorated`` as ordinary text; dual-tower
        models ignore it and consume ``raw_smiles`` structurally.
        """
        return self.encode_text(decorated, batch_size)

    def _encode_text(self, texts: list[str], batch_size: int) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        """Release weights so the next model does not have to share the device."""


def _torch_device(device: str):
    import torch

    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _free(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.empty_cache()


class SentenceTransformerEncoder(Encoder):
    """Whatever pooling the model's own sentence-transformers config specifies."""

    def __init__(self, key: str, model_name: str, device: str, label: str = ""):
        self.key = key
        self.model_name = model_name
        self.label = label or model_name
        self._device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            dev = None if self._device == "auto" else self._device
            self._model = SentenceTransformer(self.model_name, device=dev)
        return self._model

    def _encode_text(self, texts: list[str], batch_size: int) -> np.ndarray:
        model = self._load()
        return np.asarray(
            model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float64,
        )

    def close(self) -> None:
        self._model = None


class LastTokenEncoder(Encoder):
    """Last token of the last hidden state — Qwen3-Embedding's native pooling.

    Left padding puts the final real token at index -1 for every row, which is
    how Qwen's own examples do it and avoids gathering per-row lengths.
    """

    def __init__(
        self, key: str, model_name: str, device: str, max_length: int = 512, label: str = ""
    ):
        self.key = key
        self.model_name = model_name
        self.label = label or f"{model_name} (last-token)"
        self._device_arg = device
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._device = _torch_device(self._device_arg)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, padding_side="left"
            )
            self._model = (
                AutoModel.from_pretrained(self.model_name, torch_dtype=torch.float32)
                .to(self._device)
                .eval()
            )
        return self._model

    def _encode_text(self, texts: list[str], batch_size: int) -> np.ndarray:
        import torch

        model = self._load()
        out: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                hidden = model(**enc).last_hidden_state
            out.append(hidden[:, -1, :].float().cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float64)

    def close(self) -> None:
        if self._device is not None:
            self._model = None
            self._tokenizer = None
            _free(self._device)


class T5EncoderMeanPoolEncoder(Encoder):
    """T5 encoder hidden states, mean-pooled over non-pad tokens (MolT5)."""

    def __init__(
        self, key: str, model_name: str, device: str, max_length: int = 512, label: str = ""
    ):
        self.key = key
        self.model_name = model_name
        self.label = label or f"{model_name} (encoder, mean-pool)"
        self._device_arg = device
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoTokenizer, T5EncoderModel

            self._device = _torch_device(self._device_arg)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = (
                T5EncoderModel.from_pretrained(self.model_name, torch_dtype=torch.float32)
                .to(self._device)
                .eval()
            )
        return self._model

    def _encode_text(self, texts: list[str], batch_size: int) -> np.ndarray:
        import torch

        model = self._load()
        out: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            out.append(pooled.float().cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float64)

    def close(self) -> None:
        if self._device is not None:
            self._model = None
            self._tokenizer = None
            _free(self._device)


class MoleculeSTMEncoder(Encoder):
    """MoleculeSTM's contrastively-aligned GIN + SciBERT towers.

    Mirrors ``demos/demo_downstream_retrieval_Graph.ipynb`` exactly: SciBERT
    ``pooler_output`` → ``text2latent`` for text, 5-layer GIN mean-pooled →
    ``mol2latent`` for molecules, both into the shared 256-d space. Deviating
    from the demo's pooling or projections would produce embeddings that look
    plausible and mean nothing, so this follows the reference implementation
    rather than reimplementing it.

    The SMILES branch of MoleculeSTM is MegaMolBART, which needs ``megatron``
    and ``apex``; the Graph branch needs only ``torch_geometric`` + ``ogb``, so
    that is the one used here. Both towers were trained together, so this is the
    published model either way.

    Cosines here can be negative, unlike the single-tower models: nothing ties
    the two towers' outputs to a shared cone, only their relative ordering was
    trained. Read the gap against the mismatched control, not the absolute value.
    """

    key = "moleculestm"
    dual_tower = True

    SSL_EMB_DIM = 256
    GNN_EMB_DIM = 300
    NUM_LAYER = 5
    TEXT_MODEL = "allenai/scibert_scivocab_uncased"
    TEXT_DIM = 768
    MAX_SEQ_LEN = 512

    def __init__(self, checkpoint_dir: str, device: str, label: str = ""):
        self.checkpoint_dir = checkpoint_dir
        self.label = label or f"MoleculeSTM Graph ({os.path.basename(checkpoint_dir)})"
        self._device_arg = device
        self._loaded = False
        self._device = None

    def _load(self) -> None:
        if self._loaded:
            return
        import torch
        import torch.nn as nn
        from transformers import AutoModel, AutoTokenizer

        from MoleculeSTM.models import GNN, GNN_graphpred

        self._device = _torch_device(self._device_arg)
        self._torch = torch

        def load_into(module, filename):
            path = os.path.join(self.checkpoint_dir, filename)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"{path} not found. Download the MoleculeSTM checkpoints with:\n"
                    f"    {MOLECULESTM_DOWNLOAD_HINT}"
                )
            state = torch.load(path, map_location="cpu")
            # These were persistent buffers in the transformers version MoleculeSTM
            # was saved with and are derived constants now. Dropping them by name
            # keeps the load strict, so a genuinely mismatched checkpoint still
            # raises instead of silently leaving weights at their init values.
            dropped = [k for k in _STALE_BUFFER_KEYS if k in state]
            for key in dropped:
                del state[key]
            if dropped:
                print(
                    f"    {filename}: dropped stale buffer(s) {', '.join(dropped)}",
                    flush=True,
                )
            module.load_state_dict(state)
            return module

        self._tokenizer = AutoTokenizer.from_pretrained(self.TEXT_MODEL)
        self._text_model = load_into(
            AutoModel.from_pretrained(self.TEXT_MODEL), "text_model.pth"
        ).to(self._device).eval()
        self._text2latent = load_into(
            nn.Linear(self.TEXT_DIM, self.SSL_EMB_DIM), "text2latent_model.pth"
        ).to(self._device).eval()

        node_model = GNN(
            num_layer=self.NUM_LAYER,
            emb_dim=self.GNN_EMB_DIM,
            JK="last",
            drop_ratio=0.0,
            gnn_type="gin",
        )
        self._mol_model = load_into(
            GNN_graphpred(
                num_layer=self.NUM_LAYER,
                emb_dim=self.GNN_EMB_DIM,
                JK="last",
                graph_pooling="mean",
                num_tasks=1,
                molecule_node_model=node_model,
            ),
            "molecule_model.pth",
        ).to(self._device).eval()
        self._mol2latent = load_into(
            nn.Linear(self.GNN_EMB_DIM, self.SSL_EMB_DIM), "mol2latent_model.pth"
        ).to(self._device).eval()
        self._loaded = True

    def _encode_text(self, texts: list[str], batch_size: int) -> np.ndarray:
        self._load()
        torch = self._torch
        out: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.MAX_SEQ_LEN,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                pooled = self._text_model(**enc)["pooler_output"]
                out.append(self._text2latent(pooled).float().cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float64)

    def encode_molecules(
        self, decorated: Sequence[str], raw_smiles: Sequence[str], batch_size: int
    ) -> np.ndarray:
        self._load()
        torch = self._torch
        from rdkit import Chem
        from torch_geometric.loader import DataLoader as PyGDataLoader

        from MoleculeSTM.datasets.utils import mol_to_graph_data_obj_simple

        graphs = []
        for smiles in raw_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"MoleculeSTM: unparseable SMILES {smiles!r}")
            graphs.append(mol_to_graph_data_obj_simple(mol))

        out: list[np.ndarray] = []
        for batch in PyGDataLoader(graphs, batch_size=batch_size, shuffle=False):
            batch = batch.to(self._device)
            with torch.no_grad():
                repr_, _ = self._mol_model(batch)
                out.append(self._mol2latent(repr_).float().cpu().numpy())
        return _l2_normalize(np.concatenate(out, axis=0).astype(np.float64))

    def close(self) -> None:
        if self._loaded:
            self._text_model = self._mol_model = None
            self._text2latent = self._mol2latent = None
            self._loaded = False
            _free(self._device)


@dataclass
class EncoderSpec:
    key: str
    build: Callable[[argparse.Namespace, str], Encoder]
    default_on: bool = True
    requires: tuple[str, ...] = field(default_factory=tuple)
    install_hint: str = ""


ENCODER_SPECS: tuple[EncoderSpec, ...] = (
    EncoderSpec(
        "chemate",
        lambda a, dev: SentenceTransformerEncoder(
            "chemate", a.chemate_model, dev, label=f"CheMatE ({a.chemate_model})"
        ),
        requires=("sentence_transformers",),
        install_hint="pip install sentence-transformers",
    ),
    EncoderSpec(
        "molt5",
        lambda a, dev: T5EncoderMeanPoolEncoder(
            "molt5", a.molt5_model, dev, label=f"MolT5 ({a.molt5_model})"
        ),
        requires=("transformers",),
        install_hint="pip install transformers",
    ),
    EncoderSpec(
        "moleculestm",
        lambda a, dev: MoleculeSTMEncoder(a.moleculestm_dir, dev),
        requires=("torch_geometric", "ogb", "MoleculeSTM"),
        # ogb must stay <= 1.3.5: 1.3.6 appended a 'misc' bucket to the chirality
        # feature, which both resizes the checkpoint's atom embedding table and
        # shifts the feature indices it was trained on. torch_scatter needs
        # --no-build-isolation because it imports torch at build time.
        install_hint=(
            "pip install torch_geometric 'ogb==1.3.5' && "
            "pip install --no-build-isolation torch_scatter && "
            "pip install git+https://github.com/chao1224/MoleculeSTM.git"
        ),
    ),
    EncoderSpec(
        "qwen3",
        lambda a, dev: LastTokenEncoder(
            "qwen3", a.qwen3_model, dev, label=f"Qwen3 ({a.qwen3_model}, last-token)"
        ),
        requires=("transformers",),
        install_hint="pip install transformers",
    ),
    EncoderSpec(
        "qwen3-st",
        lambda a, dev: SentenceTransformerEncoder(
            "qwen3-st", a.qwen3_model, dev, label=f"Qwen3 ({a.qwen3_model}, ST)"
        ),
        default_on=False,
        requires=("sentence_transformers",),
        install_hint="pip install sentence-transformers",
    ),
)

ENCODER_KEYS = tuple(spec.key for spec in ENCODER_SPECS)
DEFAULT_ENCODER_KEYS = tuple(spec.key for spec in ENCODER_SPECS if spec.default_on)


def missing_requirements(spec: EncoderSpec) -> list[str]:
    import importlib.util

    missing = []
    for module in spec.requires:
        try:
            if importlib.util.find_spec(module) is None:
                missing.append(module)
        except (ImportError, ValueError):
            missing.append(module)
    return missing


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


def retrieval_metrics(sims: np.ndarray) -> dict[str, float]:
    """Recall@k / MRR for row-query against column-corpus, true match on the diagonal."""
    n = sims.shape[0]
    order = np.argsort(-sims, axis=1)
    ranks = np.argmax(order == np.arange(n)[:, None], axis=1) + 1
    return {
        **{f"recall_at_{k}": float(np.mean(ranks <= k)) for k in RECALL_KS},
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "chance_recall_at_1": 1.0 / n,
    }


def alignment_metrics(mol_emb: np.ndarray, text_emb: np.ndarray) -> dict:
    """Matched-vs-mismatched cosine and retrieval, both directions.

    ``sims[i, j]`` is description *i* against molecule *j*, so the diagonal holds
    matched pairs and everything else is the mismatched control.
    """
    sims = text_emb @ mol_emb.T
    n = sims.shape[0]
    matched = np.diag(sims).copy()
    off_diagonal = sims[~np.eye(n, dtype=bool)]

    pooled_std = float(np.sqrt((matched.std() ** 2 + off_diagonal.std() ** 2) / 2.0))
    return {
        "n": int(n),
        "matched": summarize(matched),
        "random_pair": summarize(off_diagonal),
        "gap_matched_minus_random": float(matched.mean() - off_diagonal.mean()),
        "cohens_d": (
            float((matched.mean() - off_diagonal.mean()) / pooled_std)
            if pooled_std > 0
            else 0.0
        ),
        # The molecule-recall direction: given a description, find its molecule.
        "retrieval": retrieval_metrics(sims),
        "retrieval_molecule_to_text": retrieval_metrics(sims.T),
        "_matched_values": matched,
        "_random_values": off_diagonal,
    }


def structure_agreement(
    emb: np.ndarray, smiles: Sequence[str], pairs: Sequence[tuple[int, int]]
) -> dict:
    """Embedding cosine vs. Morgan/Tanimoto over the given molecule pairs."""
    cos_vals = np.asarray([float(emb[i] @ emb[j]) for i, j in pairs])
    tfs_vals = np.asarray([tanimoto_similarity(smiles[i], smiles[j]) for i, j in pairs])
    out: dict = {
        "n_pairs": int(cos_vals.size),
        "embed_cosine": summarize(cos_vals),
        "tanimoto": summarize(tfs_vals),
        "_cos_values": cos_vals,
        "_tfs_values": tfs_vals,
    }
    if cos_vals.std() > 0 and tfs_vals.std() > 0:
        out["pearson_r"] = float(np.corrcoef(cos_vals, tfs_vals)[0, 1])
        try:
            from scipy.stats import spearmanr

            out["spearman_rho"] = float(spearmanr(cos_vals, tfs_vals).statistic)
        except ImportError:
            pass
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Experiment
# ─────────────────────────────────────────────────────────────────────────────


def load_pairs(path: str, n: int, seed: int) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            target = str(row.get("target") or row.get("word") or "").strip()
            definition = str(row.get("definition") or "").strip()
            if target and definition:
                rows.append((target, definition))
    if not rows:
        raise ValueError(f"{path}: no (target, definition) pairs found")
    rnd = random.Random(seed)
    if n and n < len(rows):
        rows = rnd.sample(rows, n)
    return rows


def sample_structure_pairs(
    n_molecules: int, n_pairs: int, seed: int
) -> list[tuple[int, int]]:
    """Distinct random index pairs, drawn once so every model sees the same set."""
    rnd = random.Random(seed)
    pairs: list[tuple[int, int]] = []
    while len(pairs) < n_pairs:
        i, j = rnd.randrange(n_molecules), rnd.randrange(n_molecules)
        if i != j:
            pairs.append((i, j))
    return pairs


def run_encoder(
    encoder: Encoder,
    pairs: Sequence[tuple[str, str]],
    structure_pairs: Sequence[tuple[int, int]],
    batch_size: int,
) -> dict:
    """All variants plus structure agreement for one model.

    Each distinct string set is encoded once and reused across variants, which is
    what keeps a five-model run affordable.
    """
    smiles = [s for s, _ in pairs]
    definitions = [d for _, d in pairs]

    cache: dict[tuple[str, str], np.ndarray] = {}

    def embed(side: str, treatment: str) -> np.ndarray:
        if (side, treatment) not in cache:
            if side == "mol":
                texts = [molecule_text(s, treatment) for s in smiles]
                print(f"    encoding {treatment} ({len(texts)} molecules)...", flush=True)
                cache[(side, treatment)] = encoder.encode_molecules(
                    texts, smiles, batch_size
                )
            else:
                texts = [
                    definition_text(s, d, treatment) for s, d in zip(smiles, definitions)
                ]
                print(f"    encoding {treatment} ({len(texts)} texts)...", flush=True)
                cache[(side, treatment)] = encoder.encode_text(texts, batch_size)
        return cache[(side, treatment)]

    # A dual-tower model reads a graph on the molecule side, so text decoration
    # there is a no-op: fold those variants onto the bare molecule and drop the
    # ones that then duplicate an earlier row. The definition-side treatments stay
    # meaningful either way, since the text tower does read text.
    def effective_mol(treatment: str) -> str:
        return MOL_BARE if encoder.dual_tower else treatment

    variants: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    for variant in VARIANTS:
        mol_treatment = effective_mol(variant.mol)
        name = variant_name(mol_treatment, variant.defn)
        if name in variants:
            skipped[variant.name] = (
                f"dual-tower model: molecule-side decoration is a no-op, so this "
                f"is identical to {name}"
            )
            continue
        result = alignment_metrics(
            embed("mol", mol_treatment), embed("defn", variant.defn)
        )
        result["question"] = variant.question
        variants[name] = result

    # Does this model's geometry track structure at all? Asked of the molecule text
    # that embed_sim scores and of the definition text --use-definition-embeds
    # trains on, because the losses operate in the latter space.
    structure: dict[str, dict] = {}
    for treatment, side in (
        (effective_mol(MOL_PROMPTED), "mol"),
        (DEFN_PROMPTED, "defn"),
    ):
        structure[treatment] = structure_agreement(
            embed(side, treatment), smiles, structure_pairs
        )

    return {"variants": variants, "skipped_variants": skipped, "structure": structure}


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


def strip_arrays(obj: dict) -> dict:
    return {
        k: (strip_arrays(v) if isinstance(v, dict) else v)
        for k, v in obj.items()
        if not k.startswith("_")
    }


def print_report(results: dict[str, dict], labels: dict[str, str]) -> None:
    print("\n" + "=" * 100)
    print("SMILES ↔ description alignment on ChEBI-20")
    print("=" * 100)
    for key, label in labels.items():
        print(f"  {key:<14} {label}")

    header = (
        f"\n{'model':<14}{'variant':<42}{'matched':>9}{'random':>9}{'gap':>9}"
        f"{'d':>6}{'R@1':>8}{'R@10':>8}{'MRR':>8}{'medR':>7}"
    )
    print(header)
    print("-" * len(header.strip("\n")))
    for key, res in results.items():
        for name, v in res["variants"].items():
            r = v["retrieval"]
            print(
                f"{key:<14}{name:<42}{v['matched']['mean']:>9.4f}"
                f"{v['random_pair']['mean']:>9.4f}"
                f"{v['gap_matched_minus_random']:>+9.4f}{v['cohens_d']:>6.2f}"
                f"{r['recall_at_1']:>8.4f}{r['recall_at_10']:>8.4f}"
                f"{r['mrr']:>8.4f}{r['median_rank']:>7.0f}"
            )
        for name, why in res["skipped_variants"].items():
            print(f"{key:<14}{name:<42}{'— skipped: ' + why[:44]:>55}")

    chance = chance_recall_at_1(results)
    if chance is not None:
        print(f"\nchance Recall@1 = {chance:.2e}  (retrieval is description → molecule)")

    print("\nEmbedding cosine vs. Morgan/Tanimoto on random molecule pairs:")
    print(
        f"{'model':<14}{'text':<26}{'cos mean':>10}{'cos p05':>9}{'cos p95':>9}"
        f"{'tfs mean':>10}{'r':>7}{'rho':>7}"
    )
    for key, res in results.items():
        for treatment, s in res["structure"].items():
            print(
                f"{key:<14}{treatment:<26}{s['embed_cosine']['mean']:>10.4f}"
                f"{s['embed_cosine']['p05']:>9.4f}{s['embed_cosine']['p95']:>9.4f}"
                f"{s['tanimoto']['mean']:>10.4f}"
                f"{s.get('pearson_r', float('nan')):>7.2f}"
                f"{s.get('spearman_rho', float('nan')):>7.2f}"
            )
    print("=" * 100 + "\n")


def _style(ax) -> None:
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def ordered_variant_names(results: dict[str, dict]) -> list[str]:
    """Union of variant names across models, in first-appearance order."""
    names: list[str] = []
    for res in results.values():
        for name in res["variants"]:
            if name not in names:
                names.append(name)
    return names


def chance_recall_at_1(results: dict[str, dict]) -> Optional[float]:
    for res in results.values():
        for v in res["variants"].values():
            return v["retrieval"]["chance_recall_at_1"]
    return None


def plot_retrieval(results: dict[str, dict], save_path: str) -> Optional[str]:
    """Headline figure: can a description retrieve its own molecule, per model?"""
    plt = _plt()
    if plt is None:
        return None

    names = ordered_variant_names(results)
    chance = chance_recall_at_1(results)
    if not names or chance is None:
        return None

    models = list(results)
    fig, axes = plt.subplots(1, len(RECALL_KS), figsize=(6.0 * len(RECALL_KS), 4.8))

    # Log scale: the leakage variant runs two orders of magnitude above the honest
    # ones, which on a linear axis flattens every row that actually matters into
    # the baseline. Bars are floored just under chance so a zero still draws.
    floor = chance / 3.0

    for ax, k in zip(np.atleast_1d(axes), RECALL_KS):
        width = 0.8 / max(len(names), 1)
        xs = np.arange(len(models))
        for i, name in enumerate(names):
            values = [
                results[m]["variants"].get(name, {}).get("retrieval", {}).get(
                    f"recall_at_{k}", np.nan
                )
                for m in models
            ]
            drawn = [floor if (v is not None and v <= floor) else v for v in values]
            ax.bar(xs + i * width, drawn, width=width, bottom=floor, label=name)
        ax.axhline(
            chance, color="0.2", ls="--", lw=1.4, label=f"chance = {chance:.1e}"
        )
        ax.set_yscale("log")
        ax.set_xticks(xs + 0.4 - width / 2)
        ax.set_xticklabels(models, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel(f"Recall@{k}  (log)", fontsize=10)
        ax.set_title(f"description → molecule, Recall@{k}", fontsize=11)
        _style(ax)
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="upper left")

    fig.suptitle(
        "Can a ChEBI description retrieve its own molecule?\n"
        "(*__defn_prompted embeds the target SMILES in the query — inflated by "
        "string overlap, not alignment)",
        fontsize=12,
        y=1.06,
    )
    fig.tight_layout()
    save_plot(fig, save_path, dpi=200)
    plt.close(fig)
    return save_path


def plot_matched_vs_random(
    results: dict[str, dict], save_path: str, prefer: Sequence[str]
) -> Optional[str]:
    """Is the matched-pair cosine distinguishable from the mismatched control?

    Each model contributes the first of ``prefer`` it actually has, so a
    dual-tower model shows its bare-molecule row rather than being dropped.
    """
    plt = _plt()
    if plt is None:
        return None

    usable: list[tuple[str, str, dict]] = []
    for key, res in results.items():
        for name in prefer:
            if name in res["variants"]:
                usable.append((key, name, res["variants"][name]))
                break
    if not usable:
        return None

    fig, axes = plt.subplots(1, len(usable), figsize=(5.2 * len(usable), 4.2))
    for ax, (key, name, v) in zip(np.atleast_1d(axes), usable):
        lo = min(v["_random_values"].min(), v["_matched_values"].min())
        hi = max(v["_random_values"].max(), v["_matched_values"].max())
        bins = np.linspace(lo, hi, 50)
        ax.hist(
            v["_random_values"],
            bins=bins,
            density=True,
            color="0.6",
            alpha=0.75,
            label="mismatched",
        )
        ax.hist(
            v["_matched_values"],
            bins=bins,
            density=True,
            color="#B2182B",
            alpha=0.75,
            label="matched",
        )
        ax.set_xlabel("cosine", fontsize=10)
        ax.set_ylabel("density", fontsize=10)
        ax.set_title(
            f"{key} — {name}\n"
            f"gap {v['gap_matched_minus_random']:+.3f}  (d = {v['cohens_d']:.2f})",
            fontsize=10,
        )
        ax.legend(fontsize=8)
        _style(ax)

    fig.suptitle(
        "Matched vs. mismatched pairs (undecorated definition)", fontsize=13, y=1.03
    )
    fig.tight_layout()
    save_plot(fig, save_path, dpi=200)
    plt.close(fig)
    return save_path


def plot_structure_agreement(results: dict[str, dict], save_path: str) -> Optional[str]:
    """Does each model's cosine track Morgan/Tanimoto on random molecule pairs?"""
    plt = _plt()
    if plt is None:
        return None

    panels = [
        (key, treatment, s)
        for key, res in results.items()
        for treatment, s in res["structure"].items()
    ]
    if not panels:
        return None

    cols = min(4, len(panels))
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.2 * rows), squeeze=False)
    flat = axes.ravel()
    for ax, (key, treatment, s) in zip(flat, panels):
        ax.scatter(
            s["_tfs_values"],
            s["_cos_values"],
            s=6,
            alpha=0.25,
            color="#2166AC",
            edgecolors="none",
        )
        corr = ", ".join(
            f"{label}={s[k]:.2f}"
            for k, label in (("pearson_r", "r"), ("spearman_rho", "ρ"))
            if k in s
        )
        ax.set_xlabel("Morgan Tanimoto (ECFP4)", fontsize=10)
        ax.set_ylabel("embedding cosine", fontsize=10)
        ax.set_title(f"{key} — {treatment}" + (f"\n{corr}" if corr else ""), fontsize=10)
        _style(ax)
    for ax in flat[len(panels) :]:
        ax.set_visible(False)

    fig.suptitle(
        "Embedding cosine vs. structural similarity on random molecule pairs",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    save_plot(fig, save_path, dpi=200)
    plt.close(fig)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--definitions",
        default=default_definitions_path(TASK),
        help=(
            "definitions.jsonl from prepare_molopt_chebi20.py — the exact "
            "(target, definition) pairs training consumes."
        ),
    )
    p.add_argument(
        "--models",
        nargs="*",
        default=list(DEFAULT_ENCODER_KEYS),
        choices=list(ENCODER_KEYS),
        help="Which encoders to run. Unavailable ones are skipped with an install hint.",
    )
    p.add_argument(
        "--n",
        type=int,
        default=1000,
        help=(
            "Molecules to sample. Retrieval chance level is 1/n, so a larger n is "
            "a strictly harder test — compare runs only at equal n."
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--n-structure-pairs",
        type=int,
        default=4000,
        help="Random molecule pairs for the cosine-vs-Tanimoto comparison.",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--device",
        default="auto",
        help="torch device for the non-sentence-transformers encoders (auto/cpu/cuda/mps).",
    )
    p.add_argument(
        "--out-dir",
        default=os.path.join("data", "molopt", "analysis"),
        help="Where the JSON report and figures are written.",
    )
    p.add_argument("--chemate-model", default="SchwallerGroup/CheMatE-v0")
    p.add_argument(
        "--molt5-model",
        default="laituan245/molt5-base",
        help=(
            "MolT5 checkpoint. Keep the pretrained base: the *-smiles2caption "
            "variants are fine-tuned on the ChEBI-20 train split, which this "
            "corpus pools, so they would be scored on their own training data."
        ),
    )
    p.add_argument("--qwen3-model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument(
        "--moleculestm-dir",
        default=MOLECULESTM_DEFAULT_DIR,
        help=(
            "MoleculeSTM Graph checkpoint dir holding text_model.pth, "
            "molecule_model.pth, text2latent_model.pth and mol2latent_model.pth. "
            f"Fetch with: {MOLECULESTM_DOWNLOAD_HINT}"
        ),
    )
    add_wandb_cli(p)
    return p.parse_args(argv)


def build_encoders(
    args: argparse.Namespace,
) -> tuple[list[Encoder], dict[str, str]]:
    encoders: list[Encoder] = []
    unavailable: dict[str, str] = {}
    by_key = {spec.key: spec for spec in ENCODER_SPECS}
    for key in args.models:
        spec = by_key[key]
        missing = missing_requirements(spec)
        if missing:
            unavailable[key] = (
                f"missing {', '.join(missing)} — install with: {spec.install_hint}"
            )
            continue
        encoders.append(spec.build(args, args.device))
    return encoders, unavailable


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not os.path.isfile(args.definitions):
        print(
            f"ERROR: {args.definitions} not found — run "
            f"scripts/prepare_molopt_chebi20.py --download first",
            file=sys.stderr,
        )
        return 1

    pairs = load_pairs(args.definitions, args.n, args.seed)
    structure_pairs = sample_structure_pairs(
        len(pairs), args.n_structure_pairs, args.seed
    )
    print(
        f"[analyze] {len(pairs)} (SMILES, description) pairs from {args.definitions}\n"
        f"[analyze] embedding_prompt      : {embedding_prompt_template(TASK)!r}\n"
        f"[analyze] embedding_prompt_defn : {definition_embedding_template(TASK)!r}"
    )

    encoders, unavailable = build_encoders(args)
    for key, why in unavailable.items():
        print(f"[analyze] SKIP {key}: {why}", file=sys.stderr)
    if not encoders:
        print("ERROR: no requested encoder is available", file=sys.stderr)
        return 1

    results: dict[str, dict] = {}
    labels: dict[str, str] = {}
    for encoder in encoders:
        print(f"\n[analyze] === {encoder.key}: {encoder.label} ===", flush=True)
        try:
            results[encoder.key] = run_encoder(
                encoder, pairs, structure_pairs, args.batch_size
            )
            labels[encoder.key] = encoder.label
        except Exception as e:  # a broken encoder must not lose the other models
            print(f"[analyze] FAILED {encoder.key}: {type(e).__name__}: {e}", file=sys.stderr)
            unavailable[encoder.key] = f"{type(e).__name__}: {e}"
        finally:
            encoder.close()

    if not results:
        print("ERROR: every encoder failed", file=sys.stderr)
        return 1

    print_report(results, labels)

    os.makedirs(args.out_dir, exist_ok=True)
    report = {
        "definitions_path": os.path.abspath(args.definitions),
        "n_pairs": len(pairs),
        "n_structure_pairs": len(structure_pairs),
        "seed": args.seed,
        "embedding_prompt": embedding_prompt_template(TASK),
        "embedding_prompt_defn": definition_embedding_template(TASK),
        "models": labels,
        "unavailable": unavailable,
        "variant_questions": {v.name: v.question for v in VARIANTS},
        "results": {k: strip_arrays(v) for k, v in results.items()},
    }
    json_path = os.path.join(args.out_dir, "embedding_models.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[analyze] wrote {json_path}")

    for path in (
        plot_retrieval(results, os.path.join(args.out_dir, "embedding_models_retrieval.png")),
        plot_matched_vs_random(
            results,
            os.path.join(args.out_dir, "embedding_models_matched_vs_random.png"),
            (
                variant_name(MOL_PROMPTED, DEFN_BARE),
                variant_name(MOL_BARE, DEFN_BARE),
            ),
        ),
        plot_structure_agreement(
            results, os.path.join(args.out_dir, "embedding_models_structure.png")
        ),
    ):
        if path:
            print(f"[analyze] wrote {path}")
    maybe_log_named_analysis(
        args.out_dir,
        json_path,
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        group=args.wandb_group,
        name=args.wandb_run_name,
        wandb_dir=args.wandb_dir,
        no_wandb=args.no_wandb,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
