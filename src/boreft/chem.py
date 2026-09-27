"""RDKit helpers for molopt: canonicalization, fingerprints, and descriptors.

SMILES are case-sensitive (``C`` is an aliphatic carbon, ``c`` an aromatic one;
``Cl``/``Br`` are single atoms), so the lower-casing that
:func:`boreft.eval.eval_suite.normalize_text` applies to text cannot be used to
compare molecule decodes. :func:`canonical_target_key` is the molopt replacement:
it maps every SMILES that denotes the same molecule to one key, so ``CCO`` and
``OCC`` count as an exact match while chemically distinct strings stay distinct.

:func:`smiles_edit_distance` is the graded version of that comparison: character
Levenshtein over canonical SMILES when both sides parse.

This module also provides the *structural* similarity signal — Morgan (ECFP)
fingerprints and Tanimoto similarity — which is reported alongside the embedding
cosine from :mod:`boreft.text_similarity`. The two answer different questions: the
embedding says "does this read like the same kind of molecule", Tanimoto says "do
these share substructures", and a learned space can score well on one and badly
on the other.

The module also owns the fixed-order physicochemical descriptor vector written to
``definitions_rdkit.jsonl``. Descriptor similarity robust-scales each dimension
with statistics from the committed molopt corpus, then applies an RBF to normalized
RMS distance. This captures coarse property similarity; it does not replace the
substructure-sensitive Tanimoto signal.

:func:`repair_smiles` is an optional post-hoc fix (SmiSelf: invalid SMILES →
SELFIES → SMILES). Training and ``validity_rate`` never call it.
:func:`maybe_repair_invalid_smiles` is the search default: valid strings stay
as generated, invalid ones are replaced when SmiSelf can parse a repair.

RDKit is imported lazily so a Semantle-only environment never pays the import
cost (or needs the dependency at all).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from functools import lru_cache
from typing import Any, Mapping, Optional, Sequence

import numpy as np

_RDKIT_CHEM = None
_MORGAN_GENERATOR = None

# ECFP4 (radius 2, 2048 bits) is the default structural fingerprint in the
# molecular-generation literature (GuacaMol, MOSES), so TFS numbers here are
# directly comparable to published baselines.
MORGAN_RADIUS = 2
MORGAN_N_BITS = 2048

# GENZ normalizes the same decodes several times over (unseen rate, per-target-set
# coverage, validity), and at default settings that is >50k strings per run, so the
# parse is memoized. Bounded because Sobol decodes are mostly distinct junk.
_CANONICAL_CACHE_SIZE = 200_000

# BioMedGPT-Mol / LlaSMol molecular-language tags. Training may wrap generation
# targets; eval unwraps before RDKit, embeddings, and exact match.
SMILES_OPEN_TAG = "<SMILES>"
SMILES_CLOSE_TAG = "</SMILES>"

# MiST continued-pretrain markup (Schwaller et al.).
# Completion: prompt ends with ``[START_SMILES]`` (after marker inject) and
# gold is ``SMILES [END_SMILES]``. Chat: gold is the full pair; the open tag
# is not in the user turn because generation starts a new assistant span.
# ``[BEGIN_SMILES]`` appears in their SCS template.
MIST_SMILES_OPEN_TAG = "[START_SMILES]"
MIST_BEGIN_SMILES_OPEN_TAG = "[BEGIN_SMILES]"
MIST_SMILES_CLOSE_TAG = "[END_SMILES]"
MIST_MOL_OPEN_TAG = "[START_MOL]"
MIST_MOL_CLOSE_TAG = "[END_MOL]"
_MIST_SMILES_OPEN_TAGS = (MIST_SMILES_OPEN_TAG, MIST_BEGIN_SMILES_OPEN_TAG)
_MIST_MOL_SPAN_RE = re.compile(
    re.escape(MIST_MOL_OPEN_TAG) + r".*?" + re.escape(MIST_MOL_CLOSE_TAG),
    re.DOTALL,
)


def unwrap_smiles_tags(text: str) -> str:
    """Return the inner SMILES if a known tag pair is present, else ``text``.

    Understands ``<SMILES>...</SMILES>`` (LlaSMol) and MiST
    ``[START_SMILES]`` / ``[BEGIN_SMILES]`` … ``[END_SMILES]``. Uses the last
    complete pair so a trailing answer wins over an earlier example in prose.
    A close tag without an open tag is treated as a generation suffix after a
    prompt that already opened the span. Incomplete or empty tags are left
    as-is.
    """
    s = str(text).strip()
    if not s:
        return s
    xml_end = s.rfind(SMILES_CLOSE_TAG)
    mist_end = s.rfind(MIST_SMILES_CLOSE_TAG)
    if xml_end < 0 and mist_end < 0:
        return s
    if mist_end > xml_end:
        inner = _unwrap_mist_smiles_span(s, mist_end)
    else:
        inner = _unwrap_xml_smiles_span(s, xml_end)
    if inner != s:
        return unwrap_smiles_tags(inner)
    return s


def _unwrap_xml_smiles_span(s: str, end: int) -> str:
    start = s.rfind(SMILES_OPEN_TAG, 0, end)
    if start == -1:
        return s
    inner = s[start + len(SMILES_OPEN_TAG) : end].strip()
    return inner if inner else s


def _strip_mist_mol_spans(text: str) -> str:
    """Drop ``[START_MOL]...[END_MOL]`` name markup; keep SMILES text."""
    return _MIST_MOL_SPAN_RE.sub("", text).strip()


def _unwrap_mist_smiles_span(s: str, end: int) -> str:
    start = -1
    open_len = 0
    for tag in _MIST_SMILES_OPEN_TAGS:
        idx = s.rfind(tag, 0, end)
        if idx > start:
            start = idx
            open_len = len(tag)
    if start == -1:
        inner = s[:end].strip()
    else:
        inner = s[start + open_len : end].strip()
    inner = _strip_mist_mol_spans(inner)
    return inner if inner else s


def wrap_smiles_tags(smiles: str) -> str:
    """Wrap a bare SMILES string in ``<SMILES>...</SMILES>`` (idempotent)."""
    inner = unwrap_smiles_tags(smiles)
    return f"{SMILES_OPEN_TAG}{inner}{SMILES_CLOSE_TAG}"


def wrap_mist_smiles_tags(smiles: str, *, include_open: bool = False) -> str:
    """MiST-tagged gold SMILES (idempotent).

    Completion gold is ``{smiles} [END_SMILES]`` because the prompt already
    ends with ``[START_SMILES]`` and training joins ``prompt + " " + target``.
    Chat gold includes the open tag: generation is a new assistant turn.
    """
    inner = unwrap_smiles_tags(smiles)
    if include_open:
        return f"{MIST_SMILES_OPEN_TAG} {inner} {MIST_SMILES_CLOSE_TAG}"
    return f"{inner} {MIST_SMILES_CLOSE_TAG}"


def append_mist_smiles_open_tag(instruction: str) -> str:
    """Append ``[START_SMILES]`` to a user/completion instruction (idempotent)."""
    text = (instruction or "").rstrip()
    if text.endswith(MIST_SMILES_OPEN_TAG):
        return text
    if not text:
        return MIST_SMILES_OPEN_TAG
    return f"{text} {MIST_SMILES_OPEN_TAG}"


def maybe_append_mist_smiles_open_tag(instruction: str, enabled: bool) -> str:
    """``append_mist_smiles_open_tag`` when ``enabled``, else ``instruction``."""
    if not enabled:
        return instruction
    return append_mist_smiles_open_tag(instruction)


def mist_open_tag_in_prompt(
    *, mist_smiles_tags: bool, use_chat_template: bool
) -> bool:
    """True when ``[START_SMILES]`` belongs at the end of the prompt.

    Call this *after* marker inject so a suffix marker is not the last token.
    Chat templates start a new assistant span, so the open tag goes in gold
    instead (``wrap_mist_smiles_tags(..., include_open=True)``).
    """
    return bool(mist_smiles_tags) and not bool(use_chat_template)


def generation_smiles(
    smiles: str,
    *,
    smiles_tags: bool = False,
    mist_smiles_tags: bool = False,
    mist_open_in_target: bool = False,
) -> str:
    """Supervised generation string: LlaSMol tags, MiST tags, or bare."""
    if smiles_tags and mist_smiles_tags:
        raise ValueError("smiles_tags and mist_smiles_tags are mutually exclusive")
    bare = unwrap_smiles_tags(smiles)
    if mist_smiles_tags:
        return wrap_mist_smiles_tags(bare, include_open=mist_open_in_target)
    if smiles_tags:
        return wrap_smiles_tags(bare)
    return bare


def first_smiles_token(text: str) -> str:
    """First whitespace-delimited token of a decode (after SMILES-tag unwrap).

    Model completions may trail a SMILES string with commentary. Validity and
    later property oracles should see only that leading candidate, which ends
    at the first space, newline, or end of text.
    """
    parts = unwrap_smiles_tags(text).split()
    return parts[0] if parts else ""


# Fixed-order on-disk schema shared by data preparation, definition formatting,
# and runtime scoring. Additions, reordering, or label-format changes require a
# version bump so embedding caches built from the textual suffix are invalidated.
RDKIT_DESCRIPTOR_SCHEMA_VERSION = 2
RDKIT_DESCRIPTOR_MAP: tuple[dict[str, Any], ...] = (
    {
        "index": 0,
        "name": "molecular_weight",
        "unit": "Da",
        "rdkit": "Descriptors.MolWt",
        "definition_label": "average molecular weight",
    },
    {
        "index": 1,
        "name": "clogp",
        "unit": None,
        "rdkit": "Descriptors.MolLogP",
        "definition_label": (
            "Wildman-Crippen octanol-water partition coefficient estimate"
        ),
    },
    {
        "index": 2,
        "name": "tpsa",
        "unit": "angstrom^2",
        "rdkit": "rdMolDescriptors.CalcTPSA",
        "definition_label": "topological polar surface area",
    },
    {
        "index": 3,
        "name": "hydrogen_bond_donors",
        "unit": "count",
        "rdkit": "rdMolDescriptors.CalcNumHBD",
        "definition_label": "number of hydrogen-bond donor groups",
    },
    {
        "index": 4,
        "name": "hydrogen_bond_acceptors",
        "unit": "count",
        "rdkit": "rdMolDescriptors.CalcNumHBA",
        "definition_label": "number of hydrogen-bond acceptor groups",
    },
    {
        "index": 5,
        "name": "rotatable_bonds",
        "unit": "count",
        "rdkit": "rdMolDescriptors.CalcNumRotatableBonds",
        "definition_label": (
            "number of rotatable bonds using RDKit's default definition"
        ),
    },
    {
        "index": 6,
        "name": "formal_charge",
        "unit": "elementary_charge",
        "rdkit": "Chem.GetFormalCharge",
        "definition_label": "net formal charge of the represented molecular graph",
    },
    {
        "index": 7,
        "name": "fraction_csp3",
        "unit": "fraction",
        "rdkit": "rdMolDescriptors.CalcFractionCSP3",
        "definition_label": "fraction of carbon atoms that are sp3-hybridized",
    },
    {
        "index": 8,
        "name": "aromatic_rings",
        "unit": "count",
        "rdkit": "rdMolDescriptors.CalcNumAromaticRings",
        "definition_label": "number of aromatic rings",
    },
    {
        "index": 9,
        "name": "atom_stereocenters",
        "unit": "count",
        "rdkit": "rdMolDescriptors.CalcNumAtomStereoCenters",
        "definition_label": (
            "number of tetrahedral atom stereocenters, specified or unspecified"
        ),
    },
)
RDKIT_DESCRIPTOR_DIM = len(RDKIT_DESCRIPTOR_MAP)
RDKIT_ROBUST_SCALE_FACTOR = 1.4826
DEFAULT_RDKIT_SIM_TAU = 0.5
_DESCRIPTOR_CACHE_SIZE = 100_000


def _chem():
    """Return ``rdkit.Chem``, silencing RDKit's parse chatter on first use."""
    global _RDKIT_CHEM
    if _RDKIT_CHEM is None:
        try:
            from rdkit import Chem, RDLogger
        except ImportError as e:  # pragma: no cover - dependency guard
            raise ImportError(
                "rdkit is required for the molopt task "
                "(pip install rdkit, or see requirements.txt)"
            ) from e
        # Invalid decodes are expected and scored, not exceptional; without this
        # every unparseable generation writes a syntax error to stderr.
        RDLogger.DisableLog("rdApp.*")
        _RDKIT_CHEM = Chem
    return _RDKIT_CHEM


def _rdkit_or_none(compute):
    """Run an RDKit call; C++ invariants score as unparseable, not fatal.

    ``MolFromSmiles`` usually returns ``None`` for junk, but ``MolToSmiles``
    (``Canon.cpp``) and some fingerprint/descriptor calls can abort with
    ``RuntimeError: Invariant Violation``. Full eval used to treat that as
    fatal and drop the whole RECON / RECON_TEST suite. Catch here so one
    pathological decode is scored as invalid.
    """
    try:
        return compute()
    except Exception:
        return None


@lru_cache(maxsize=_CANONICAL_CACHE_SIZE)
def _canonical_stripped(text: str, *, isomeric: bool = True) -> Optional[str]:
    Chem = _chem()

    def _run() -> Optional[str]:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=isomeric)

    return _rdkit_or_none(_run)


def canonical_smiles(smiles: str, *, isomeric: bool = True) -> Optional[str]:
    """Canonical SMILES for ``smiles``, or ``None`` when it does not parse.

    ``isomeric=True`` (default) keeps stereochemistry. ``isomeric=False``
    is the stereo-stripped graph so two stereoisomers compare equal.
    """
    text = unwrap_smiles_tags(smiles)
    if not text:
        return None
    return _canonical_stripped(text, isomeric=isomeric)


def is_valid_smiles(smiles: str) -> bool:
    """True when ``smiles`` parses into a molecule."""
    return canonical_smiles(smiles) is not None


def repair_smiles(smiles: str) -> Optional[str]:
    """Best-effort valid SMILES via SmiSelf (invalid SMILES → SELFIES → SMILES).

    Already-valid strings are returned as RDKit canonical SMILES without
    calling SmiSelf. Invalid strings are encoded with ``strict=False`` so
    grammar-guided token drops can run, then decoded and re-checked with
    RDKit. Returns ``None`` when SmiSelf is not installed, encoding or
    decoding fails (including SmiSelf crashes such as ``IndexError`` on
    oversized ring digits), or the result still does not parse.

    This is a repair heuristic, not a validity oracle: leftover tokens
    can become a different molecule (``banana`` → ``BNN``). Do not fold
    it into :func:`validity_rate`.
    """
    text = unwrap_smiles_tags(smiles)
    if not text:
        return None
    canonical = _canonical_stripped(text)
    if canonical is not None:
        return canonical
    try:
        import smiself
    except ImportError:
        return None
    try:
        repaired = smiself.decoder(smiself.encoder(text, strict=False))
        return _canonical_stripped(repaired)
    except Exception:
        return None


def maybe_repair_invalid_smiles(smiles: str) -> tuple[str, bool]:
    """Unwrap tags and SmiSelf-repair invalid strings.

    Valid input is returned unchanged (not canonicalized). Invalid input is
    replaced with a SmiSelf repair when that parses; otherwise the unwrapped
    original is kept. Missing SmiSelf is a no-op, not an error.
    """
    text = unwrap_smiles_tags(smiles)
    if is_valid_smiles(text):
        return text, False
    repaired = repair_smiles(text)
    if repaired is not None:
        return repaired, True
    return text, False


def _chemistry_problem_text(problem) -> str:
    """``Type: Message`` for one :func:`rdkit.Chem.DetectChemistryProblems` hit."""
    kind = problem.GetType() if hasattr(problem, "GetType") else type(problem).__name__
    message = problem.Message() if hasattr(problem, "Message") else str(problem)
    return f"{kind}: {message}"


def smiles_chemistry_problems(smiles: str) -> tuple[str, ...]:
    """Chemistry issues when a string parses only with ``sanitize=False``.

    Empty when default ``MolFromSmiles`` succeeds, or when even the unsanitized
    parse fails (syntax, not chemistry). Used by the REPL to explain why a
    candidate is invalid.
    """
    Chem = _chem()
    text = unwrap_smiles_tags(smiles)
    if not text:
        return ()
    if Chem.MolFromSmiles(text) is not None:
        return ()
    mol = Chem.MolFromSmiles(text, sanitize=False)
    if mol is None:
        return ()
    return tuple(_chemistry_problem_text(p) for p in Chem.DetectChemistryProblems(mol))


def qed_value(smiles: str) -> Optional[float]:
    """RDKit QED in ``[0, 1]``, or ``None`` when ``smiles`` does not parse."""
    Chem = _chem()
    from rdkit.Chem import QED

    text = unwrap_smiles_tags(smiles)
    if not text:
        return None
    try:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return None
        return float(QED.qed(mol))
    except Exception:
        return None


def canonical_target_key(smiles: str) -> str:
    """Exact-match key for a SMILES string.

    Valid molecules collapse to their canonical form. Unparseable strings fall
    back to whitespace-collapsed (but case-preserving) text so two different
    invalid decodes do not compare equal to each other.
    """
    canonical = canonical_smiles(smiles)
    if canonical is not None:
        return canonical
    return " ".join(unwrap_smiles_tags(smiles).split())


def validity_rate(decodes: Sequence[str]) -> float:
    """Fraction of ``decodes`` that parse as molecules (0.0 for an empty list)."""
    if not decodes:
        return 0.0
    return sum(1 for d in decodes if is_valid_smiles(d)) / len(decodes)


def _levenshtein(a: str, b: str) -> int:
    """Classic two-row Levenshtein distance (insert / delete / substitute)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def smiles_edit_distance(target: str, decode: str) -> int:
    """Character Levenshtein distance between a target SMILES and a decode.

    Both sides are unwrapped from ``<SMILES>`` tags. When both parse, the
    distance is over canonical SMILES so equivalent spellings (``CCO`` vs ``OCC``)
    score 0. When the decode does not parse, the raw unwrapped strings are
    compared, so junk generations are charged the full string-edit cost rather
    than a sentinel.
    """
    t_can = canonical_smiles(target)
    d_can = canonical_smiles(decode)
    if t_can is not None and d_can is not None:
        return _levenshtein(t_can, d_can)
    t_raw = " ".join(unwrap_smiles_tags(target).split())
    d_raw = " ".join(unwrap_smiles_tags(decode).split())
    return _levenshtein(t_raw, d_raw)


def smiles_edit_dist_per_text(
    targets: Sequence[str], generated: Sequence[str]
) -> np.ndarray:
    """Per-pair SMILES edit distance. Shape ``[N]``, parallel to the inputs."""
    pairs = list(zip(targets, generated))
    if not pairs:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(
        [float(smiles_edit_distance(t, g)) for t, g in pairs], dtype=np.float64
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fixed-order RDKit descriptors / robust-scaled RBF similarity
# ─────────────────────────────────────────────────────────────────────────────


def _descriptor_values_from_mol(mol) -> Optional[tuple[float | int, ...]]:
    from rdkit.Chem import Descriptors, rdMolDescriptors

    Chem = _chem()

    def _run() -> tuple[float | int, ...]:
        return (
            round(float(Descriptors.MolWt(mol)), 6),
            round(float(Descriptors.MolLogP(mol)), 6),
            round(float(rdMolDescriptors.CalcTPSA(mol)), 6),
            int(rdMolDescriptors.CalcNumHBD(mol)),
            int(rdMolDescriptors.CalcNumHBA(mol)),
            int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
            int(Chem.GetFormalCharge(mol)),
            round(float(rdMolDescriptors.CalcFractionCSP3(mol)), 6),
            int(rdMolDescriptors.CalcNumAromaticRings(mol)),
            int(rdMolDescriptors.CalcNumAtomStereoCenters(mol)),
        )

    return _rdkit_or_none(_run)


@lru_cache(maxsize=_DESCRIPTOR_CACHE_SIZE)
def _descriptor_values_stripped(text: str) -> Optional[tuple[float | int, ...]]:
    Chem = _chem()

    def _run() -> Optional[tuple[float | int, ...]]:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return None
        return _descriptor_values_from_mol(mol)

    return _rdkit_or_none(_run)


def rdkit_descriptor_values(smiles: str) -> Optional[list[float | int]]:
    """Ten JSON-safe descriptor scalars, or ``None`` for invalid SMILES.

    Prefers the canonical form so equivalent spellings share a cache entry.
    ``MolToSmiles`` is not always a round-trip (aromatic kekulization can emit
    a string ``MolFromSmiles`` then rejects), so a failed canonical re-parse
    scores the original mol instead of raising. The same fallback applies when
    ``MolToSmiles`` raises an RDKit invariant (``Canon.cpp``).
    """
    text = unwrap_smiles_tags(smiles)
    if not text:
        return None
    canonical = _canonical_stripped(text)
    if canonical is not None:
        values = _descriptor_values_stripped(canonical)
        if values is not None:
            return list(values)
    Chem = _chem()

    def _from_original() -> Optional[list[float | int]]:
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return None
        values = _descriptor_values_from_mol(mol)
        return list(values) if values is not None else None

    return _rdkit_or_none(_from_original)


def robust_descriptor_stats(
    vectors: Sequence[Sequence[float | int]],
) -> dict[str, Any]:
    """Fixed per-dimension median/MAD scaling metadata for descriptor vectors."""
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != RDKIT_DESCRIPTOR_DIM or x.shape[0] == 0:
        raise ValueError(
            f"descriptor vectors must have shape [N, {RDKIT_DESCRIPTOR_DIM}], "
            f"got {tuple(x.shape)}"
        )
    median = np.median(x, axis=0)
    mad = np.median(np.abs(x - median), axis=0)
    scale = RDKIT_ROBUST_SCALE_FACTOR * mad
    std = np.std(x, axis=0)
    scale = np.where(scale > 1e-12, scale, std)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return {
        "method": "median_mad",
        "population": "full_corpus",
        "count": int(x.shape[0]),
        "median": [float(v) for v in median],
        "scale": [float(v) for v in scale],
        "mad_scale_factor": RDKIT_ROBUST_SCALE_FACTOR,
        "zero_mad_fallback": "population_std_then_one",
    }


def _repo_root() -> str:
    path = os.path.abspath(__file__)
    for _ in range(8):
        path = os.path.dirname(path)
        if os.path.isfile(os.path.join(path, "pyproject.toml")):
            return path
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_rdkit_definitions_path() -> str:
    return os.path.join(
        _repo_root(), "data", "molopt", "train", "definitions_rdkit.jsonl"
    )


def default_rdkit_map_path() -> str:
    return os.path.join(
        _repo_root(), "data", "molopt", "train", "definitions_rdkit_map.json"
    )


def file_sha256(path: str) -> str:
    """Hex SHA-256 of a file, read in bounded chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=16)
def _load_rdkit_descriptor_map_cached(
    resolved: str, content_sha256: str
) -> dict[str, Any]:
    del content_sha256  # cache-key component; content is read from ``resolved``.
    with open(resolved, encoding="utf-8") as f:
        metadata = json.load(f)
    if int(metadata.get("schema_version", -1)) != RDKIT_DESCRIPTOR_SCHEMA_VERSION:
        raise ValueError(
            f"{resolved}: unsupported RDKit descriptor schema "
            f"{metadata.get('schema_version')!r}"
        )
    if int(metadata.get("definition_length", -1)) != RDKIT_DESCRIPTOR_DIM:
        raise ValueError(
            f"{resolved}: definition_length must be {RDKIT_DESCRIPTOR_DIM}"
        )
    positions = metadata.get("positions")
    expected_positions = list(RDKIT_DESCRIPTOR_MAP)
    if positions != expected_positions:
        raise ValueError(
            f"{resolved}: positions must exactly match schema version "
            f"{RDKIT_DESCRIPTOR_SCHEMA_VERSION}"
        )
    normalization = metadata.get("normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError(f"{resolved}: normalization metadata is missing")
    for key in ("median", "scale"):
        values = normalization.get(key)
        if not isinstance(values, list) or len(values) != RDKIT_DESCRIPTOR_DIM:
            raise ValueError(
                f"{resolved}: normalization.{key} must have "
                f"{RDKIT_DESCRIPTOR_DIM} values"
            )
    return metadata


def load_rdkit_descriptor_map(path: str = "") -> dict[str, Any]:
    """Load and validate map metadata, invalidating cache when content changes."""
    resolved = os.path.abspath(path or default_rdkit_map_path())
    return _load_rdkit_descriptor_map_cached(resolved, file_sha256(resolved))


def _normalization_arrays(map_path: Optional[str] = None) -> tuple[np.ndarray, np.ndarray]:
    metadata = load_rdkit_descriptor_map(map_path or "")
    normalization = metadata["normalization"]
    median = np.asarray(normalization["median"], dtype=np.float64)
    scale = np.asarray(normalization["scale"], dtype=np.float64)
    if not np.all(np.isfinite(median)) or not np.all(np.isfinite(scale)):
        raise ValueError("RDKit descriptor normalization contains non-finite values")
    if np.any(scale <= 0):
        raise ValueError("RDKit descriptor normalization scales must be positive")
    return median, scale


def normalize_rdkit_values(
    values: Sequence[float | int], *, map_path: Optional[str] = None
) -> np.ndarray:
    """Robust-scale a descriptor vector with the committed corpus statistics."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.shape != (RDKIT_DESCRIPTOR_DIM,):
        raise ValueError(
            f"RDKit descriptor must have shape ({RDKIT_DESCRIPTOR_DIM},), "
            f"got {tuple(arr.shape)}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("RDKit descriptor contains non-finite values")
    median, scale = _normalization_arrays(map_path)
    return (arr - median) / scale


def normalized_rdkit_descriptor(
    smiles: str, *, map_path: Optional[str] = None
) -> Optional[np.ndarray]:
    values = rdkit_descriptor_values(smiles)
    if values is None:
        return None
    return normalize_rdkit_values(values, map_path=map_path)


def _rbf_from_normalized(za: np.ndarray, zb: np.ndarray) -> float:
    """``exp(-0.5 × mean((z₁ − z₂)²))`` on robust-normalized descriptor vectors."""
    return float(math.exp(-0.5 * float(np.mean(np.square(za - zb)))))


def rdkit_similarity_from_values(
    a: Sequence[float | int],
    b: Sequence[float | int],
    *,
    map_path: Optional[str] = None,
) -> float:
    """RBF descriptor similarity from already-computed 10-d vectors.

    Same kernel as :func:`rdkit_similarity`, skipping the SMILES parse. Used by
    synthetic-pair analyses that never have a molecule on either side.
    """
    return _rbf_from_normalized(
        normalize_rdkit_values(a, map_path=map_path),
        normalize_rdkit_values(b, map_path=map_path),
    )


def rdkit_similarity(
    a: str, b: str, *, map_path: Optional[str] = None
) -> float:
    """Robust-scaled descriptor RBF similarity in ``[0, 1]``.

    Invalid SMILES score 0.0, matching the strict invalid-decode convention used
    by :func:`tanimoto_similarity`.
    """
    za = normalized_rdkit_descriptor(a, map_path=map_path)
    zb = normalized_rdkit_descriptor(b, map_path=map_path)
    if za is None or zb is None:
        return 0.0
    return _rbf_from_normalized(za, zb)


def rdkit_sim_per_text(
    targets: Sequence[str],
    generated: Sequence[str],
    *,
    map_path: Optional[str] = None,
) -> np.ndarray:
    pairs = list(zip(targets, generated))
    if not pairs:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(
        [rdkit_similarity(t, g, map_path=map_path) for t, g in pairs],
        dtype=np.float64,
    )


def rdkit_internal_diversity(
    smiles: Sequence[str], *, map_path: Optional[str] = None
) -> Optional[float]:
    """One minus mean pairwise descriptor similarity over valid molecules."""
    rows = [
        row
        for row in (
            normalized_rdkit_descriptor(s, map_path=map_path) for s in smiles
        )
        if row is not None
    ]
    if len(rows) < 2:
        return None
    z = np.asarray(rows, dtype=np.float64)
    sq_norm = np.sum(np.square(z), axis=1)
    sq_dist = np.maximum(
        sq_norm[:, None] + sq_norm[None, :] - 2.0 * (z @ z.T), 0.0
    )
    sims = np.exp(-0.5 * sq_dist / RDKIT_DESCRIPTOR_DIM)
    upper = sims[np.triu_indices(len(z), k=1)]
    return 1.0 - float(np.mean(upper))


# ─────────────────────────────────────────────────────────────────────────────
# Morgan fingerprints / Tanimoto similarity (TFS)
# ─────────────────────────────────────────────────────────────────────────────

# Fingerprints are heavier than canonical strings (~2 KiB of bit vector plus
# object overhead each), so this cache is an order of magnitude smaller than the
# canonicalization one. The hit rate is what matters: DIST and GENZ score the same
# target against many sample bags, and every target is fingerprinted once per bag.
_FINGERPRINT_CACHE_SIZE = 100_000


def _morgan_generator():
    """Cached Morgan fingerprint generator (ECFP4 at the module defaults)."""
    global _MORGAN_GENERATOR
    if _MORGAN_GENERATOR is None:
        _chem()  # import + silence RDKit before touching its submodules
        from rdkit.Chem import rdFingerprintGenerator

        _MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
            radius=MORGAN_RADIUS, fpSize=MORGAN_N_BITS
        )
    return _MORGAN_GENERATOR


@lru_cache(maxsize=_FINGERPRINT_CACHE_SIZE)
def _fingerprint_stripped(text: str):
    Chem = _chem()

    def _run():
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return None
        return _morgan_generator().GetFingerprint(mol)

    return _rdkit_or_none(_run)


def morgan_fingerprint(smiles: str):
    """ECFP4 bit vector for ``smiles``, or ``None`` when it does not parse.

    Fingerprints are invariant to SMILES spelling, so no canonicalization is
    needed first.
    """
    text = unwrap_smiles_tags(smiles)
    if not text:
        return None
    return _fingerprint_stripped(text)


def tanimoto_similarity(a: str, b: str) -> float:
    """Morgan/Tanimoto similarity in ``[0, 1]``; 0.0 if either side is invalid.

    Scoring an unparseable decode as 0.0 makes TFS a strict penalty for invalid
    generations, the same convention :func:`validity_rate` measures separately.
    """
    fp_a = morgan_fingerprint(a)
    fp_b = morgan_fingerprint(b)
    if fp_a is None or fp_b is None:
        return 0.0
    from rdkit import DataStructs

    sim = _rdkit_or_none(lambda: float(DataStructs.TanimotoSimilarity(fp_a, fp_b)))
    return 0.0 if sim is None else sim


def tanimoto_sim_per_text(
    targets: Sequence[str], generated: Sequence[str]
) -> np.ndarray:
    """Per-pair Morgan/Tanimoto similarity. Shape ``[N]``, parallel to the inputs."""
    pairs = list(zip(targets, generated))
    if not pairs:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(
        [tanimoto_similarity(t, g) for t, g in pairs], dtype=np.float64
    )


def tanimoto_internal_diversity(smiles: Sequence[str]) -> Optional[float]:
    """Mean pairwise ``1 - Tanimoto`` over a set of molecules (GuacaMol IntDiv).

    The structural counterpart of
    :func:`boreft.eval.eval_suite.semantic_dispersion`: higher means the set
    covers more distinct chemistry. Unparseable entries are dropped rather than
    scored as 0.0, since a diversity number inflated by junk would be misleading.
    Returns ``None`` when fewer than two molecules parse, so callers can skip the
    metric instead of logging a meaningless value.
    """
    fps = [fp for fp in (morgan_fingerprint(s) for s in smiles) if fp is not None]
    if len(fps) < 2:
        return None
    from rdkit import DataStructs

    total = 0.0
    n_pairs = 0
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        total += float(sum(sims))
        n_pairs += len(sims)
    return 1.0 - (total / n_pairs)
