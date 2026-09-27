"""Black-box molecular property oracles for molopt search and screening.

TDC fingerprint classifiers (DRD2, GSK3B, JNK3) map a SMILES string to a
class-1 probability in ``[0, 1]``. Search also accepts the dual-kinase
product ``GSK3B_JNK3`` (TDC JNK3×GSK3β). Invalid molecules score
``INVALID_SCORE`` (0) rather than being dropped, so a search budget always
records an observation. The p90 train split and oracle screens still use
the three TDC names only.

sklearn 1.4+ dropped leaf-count normalization in
``DecisionTreeClassifier.predict_proba``. TDC's GSK3B/JNK3 pickles store
weighted class counts in ``tree_.value``, so without the shim the forest
returns mean leaf counts (often ``> 1``) instead of probabilities.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
from typing import Callable, Iterator, Mapping, Protocol, Sequence

import numpy as np

from boreft.chem import canonical_smiles, qed_value
from boreft.task_config import task_instruction

TDC_ORACLE_NAMES: tuple[str, ...] = ("DRD2", "GSK3B", "JNK3")
DUAL_KINASE_ORACLE = "GSK3B_JNK3"
DUAL_KINASE_FACTORS: tuple[str, ...] = ("GSK3B", "JNK3")
MOLOPT_SEARCH_ORACLE_NAMES: tuple[str, ...] = TDC_ORACLE_NAMES + (DUAL_KINASE_ORACLE,)
INVALID_SCORE = 0.0
SCORE_THRESHOLDS: tuple[float, ...] = (0.5, 0.7, 0.9)
_ORACLE_ALIASES = {
    "GSK3BETA": "GSK3B",
    "GSK3Β": "GSK3B",
    "GSK3β": "GSK3B",
}
_DUAL_KINASE_KEYS = frozenset(
    {
        DUAL_KINASE_ORACLE,
        "JNK3_GSK3B",
        "GSK3BJNK3",
        "JNK3GSK3B",
        "DUAL_KINASE",
        "DUALKINASE",
    }
)
# Same completion prefix as MiST / BOReFT, plus one objective line per oracle.
PROPERTY_SEARCH_OBJECTIVE_LINES: dict[str, str] = {
    "DRD2": "The task is to optimize for DRD2 binding.",
    "GSK3B": "The task is to optimize for GSK3β (GSK3B) inhibition.",
    "JNK3": "The task is to optimize for JNK3 inhibition.",
    DUAL_KINASE_ORACLE: (
        "The task is to optimize the product of GSK3β (GSK3B) and JNK3 inhibition."
    ),
}
PROPERTY_SEARCH_COMPLETION_PREFIX = task_instruction(
    "molopt", use_chat_template=False
)
PROPERTY_SEARCH_TASK_DESCRIPTIONS: dict[str, str] = {
    name: f"{PROPERTY_SEARCH_OBJECTIVE_LINES[name]}\n{PROPERTY_SEARCH_COMPLETION_PREFIX}"
    for name in MOLOPT_SEARCH_ORACLE_NAMES
}
# First molecule in the TDC GSK3B/JNK3 docs (scores 0.03 / 0.01).
TDC_SKLEARN_PROBE_SMILES = (
    "CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1"
)
TDC_PROBE_MIN_SCORE: dict[str, float] = {"GSK3B": 0.02, "JNK3": 0.005}
# TDC downloads pickle checkpoints to ``./oracle`` (cwd-relative; the sklearn
# loaders hardcode that name). Parallel screen jobs race on os.mkdir.
TDC_ORACLE_DIR = "oracle"

OracleFn = Callable[[Sequence[str]], Sequence[float]]


class SupportsOracle(Protocol):
    def __call__(self, smiles: Sequence[str]) -> Sequence[float]: ...


def normalize_tdc_oracle_name(name: str) -> str:
    """Map a CLI/paper alias onto one of ``TDC_ORACLE_NAMES``."""
    raw = str(name).strip()
    if not raw:
        raise ValueError("oracle name must not be blank")
    folded = raw.replace("β", "B").replace("Β", "B")
    key = _ORACLE_ALIASES.get(folded, folded).upper()
    key = _ORACLE_ALIASES.get(key, key)
    if key not in TDC_ORACLE_NAMES:
        raise ValueError(
            f"unknown TDC oracle {name!r}; expected one of {TDC_ORACLE_NAMES}"
        )
    return key


def _fold_property_oracle_key(name: str) -> str:
    folded = str(name).strip().replace("β", "B").replace("Β", "B")
    folded = folded.replace("*", "_").replace("-", "_").replace(" ", "_")
    folded = folded.upper().replace("GSK3BETA", "GSK3B")
    while "__" in folded:
        folded = folded.replace("__", "_")
    return folded.strip("_")


def normalize_property_oracle_name(name: str) -> str:
    """Map a CLI alias onto a molopt search oracle, including ``GSK3B_JNK3``."""
    raw = str(name).strip()
    if not raw:
        raise ValueError("oracle name must not be blank")
    folded = _fold_property_oracle_key(raw)
    if folded in _DUAL_KINASE_KEYS:
        return DUAL_KINASE_ORACLE
    try:
        return normalize_tdc_oracle_name(name)
    except ValueError as exc:
        raise ValueError(
            f"unknown property oracle {name!r}; expected one of "
            f"{MOLOPT_SEARCH_ORACLE_NAMES}"
        ) from exc


def property_oracle_factors(oracle: str) -> tuple[str, ...]:
    """TDC oracles that combine into this search score (one name, or GSK3B and JNK3)."""
    name = normalize_property_oracle_name(oracle)
    if name == DUAL_KINASE_ORACLE:
        return DUAL_KINASE_FACTORS
    return (name,)


def property_search_task_description(oracle: str) -> str:
    """Objective line plus the MiST completion prefix used by BOReFT."""
    name = normalize_property_oracle_name(oracle)
    return PROPERTY_SEARCH_TASK_DESCRIPTIONS[name]


@dataclass(frozen=True)
class MoleculeScore:
    smiles: str
    canonical: str | None
    valid: bool
    score: float
    qed: float | None = None


def prepare_tdc_oracle_dir(path: str = TDC_ORACLE_DIR) -> str:
    """Create TDC's checkpoint directory, or fail clearly if a file is in the way.

    PyTDC does ``if not os.path.exists(path): os.mkdir(path)``. That races when
    several screen jobs share a cwd, and it also fails when ``path`` is a file.
    """
    resolved = os.path.abspath(path)
    if os.path.isfile(resolved):
        raise RuntimeError(
            f"TDC writes oracle checkpoints under {resolved}, but that path "
            "is a file. Remove or rename it and retry."
        )
    os.makedirs(resolved, exist_ok=True)
    return resolved


@contextmanager
def _tdc_mkdir_exist_ok() -> Iterator[None]:
    """Treat ``os.mkdir`` of an existing directory as success (TDC race)."""
    real_mkdir = os.mkdir

    def mkdir(target, *args, **kwargs):
        try:
            real_mkdir(target, *args, **kwargs)
        except FileExistsError:
            if not os.path.isdir(target):
                raise

    os.mkdir = mkdir  # type: ignore[assignment]
    try:
        yield
    finally:
        os.mkdir = real_mkdir


_LEGACY_TREE_NODE_NAMES: tuple[str, ...] = (
    "left_child",
    "right_child",
    "feature",
    "threshold",
    "impurity",
    "n_node_samples",
    "weighted_n_node_samples",
)


def _upgrade_sklearn_tree_nodes(nodes: np.ndarray) -> np.ndarray:
    """Add sklearn 1.3+ ``missing_go_to_left`` so TDC RF pickles can unpickle.

    JNK3/GSK3B were pickled before missing-value routing existed. sklearn 1.3+
    rejects that node dtype; filling the new field with 0 preserves the original
    (complete-data) splits.
    """
    names = getattr(getattr(nodes, "dtype", None), "names", None)
    if not names or "missing_go_to_left" in names:
        return nodes
    if tuple(names) != _LEGACY_TREE_NODE_NAMES:
        return nodes
    try:
        from sklearn.tree._tree import NODE_DTYPE
    except ImportError:
        return nodes
    upgraded = np.empty(nodes.shape, dtype=NODE_DTYPE)
    for name in names:
        upgraded[name] = nodes[name]
    upgraded["missing_go_to_left"] = np.uint8(0)
    return upgraded


_tree_pickle_compat_installed = False
_original_check_node_ndarray = None
_original_pickle_load = None
_original_base_forest_tags = None
_original_dt_predict_proba = None


def _upgrade_sklearn_forest(obj):
    """Fill ``estimator`` on pre-1.4 sklearn forest pickles (had ``base_estimator``)."""
    if obj is None:
        return obj
    name = type(obj).__name__
    if name not in {
        "RandomForestClassifier",
        "RandomForestRegressor",
        "ExtraTreesClassifier",
        "ExtraTreesRegressor",
    }:
        return obj
    proto = getattr(obj, "estimator", None)
    if proto is None or proto == "deprecated":
        proto = getattr(obj, "base_estimator", None)
    if proto is None or proto == "deprecated":
        proto = getattr(obj, "base_estimator_", None)
    if proto is None or proto == "deprecated":
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        criterion = getattr(obj, "criterion", "gini")
        if "Regressor" in name:
            proto = DecisionTreeRegressor(criterion=criterion)
        else:
            proto = DecisionTreeClassifier(criterion=criterion)
    obj.estimator = proto
    return obj


def _row_normalize_predict_proba(proba):
    """Turn leaf class counts (or fractions) into rows that sum to 1.

    sklearn 1.4+ ``DecisionTreeClassifier.predict_proba`` returns
    ``tree_.predict`` unnormalized. Newly trained trees already store
    fractions, so this is a no-op for them; TDC GSK3B/JNK3 pickles store
    weighted sample counts and need the division.
    """
    if isinstance(proba, list):
        return [_row_normalize_predict_proba(item) for item in proba]
    arr = np.asarray(proba, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        total = float(arr.sum())
        if total <= 0.0:
            return arr
        return arr / total
    denom = arr.sum(axis=1, keepdims=True)
    np.putmask(denom, denom == 0.0, 1.0)
    return arr / denom


def _install_tree_predict_proba_normalization() -> None:
    """Restore pre-1.4 leaf-count normalization on DecisionTreeClassifier."""
    global _original_dt_predict_proba
    if _original_dt_predict_proba is not None:
        return
    try:
        from sklearn.tree import DecisionTreeClassifier
    except ImportError:
        return

    _original_dt_predict_proba = DecisionTreeClassifier.predict_proba

    def predict_proba(self, X, check_input=True):
        return _row_normalize_predict_proba(
            _original_dt_predict_proba(self, X, check_input=check_input)
        )

    DecisionTreeClassifier.predict_proba = predict_proba


def install_sklearn_legacy_tree_pickle_compat() -> bool:
    """Leave TDC RF pickle shims installed for this process.

    Three sklearn breaks show up on TDC GSK3B/JNK3 pickles:
    1. Tree node dtype (missing ``missing_go_to_left``).
    2. Forest template attr renamed ``base_estimator`` → ``estimator``.
    3. ``predict_proba`` no longer converts leaf class counts to fractions
       (sklearn 1.4+), so scores explode above 1.

    GSK3B pickles on the first *score*, not at ``Oracle()`` construction, so
    the shim must stay installed. TDC swallows score errors as 0.0.
    """
    global _tree_pickle_compat_installed
    global _original_check_node_ndarray, _original_pickle_load
    global _original_base_forest_tags
    if _tree_pickle_compat_installed:
        return True
    try:
        import pickle

        import sklearn.tree._tree as tree_mod
        from sklearn.tree._tree import NODE_DTYPE
    except ImportError:
        return False
    node_names = getattr(NODE_DTYPE, "names", None) or ()
    if "missing_go_to_left" in node_names:
        real_check = tree_mod._check_node_ndarray

        def _check_node_ndarray(node_ndarray, expected_dtype):
            return real_check(
                _upgrade_sklearn_tree_nodes(node_ndarray), expected_dtype
            )

        _original_check_node_ndarray = real_check
        tree_mod._check_node_ndarray = _check_node_ndarray
    if _original_pickle_load is None:
        _original_pickle_load = pickle.load

        def _pickle_load(file, *args, **kwargs):
            obj = _original_pickle_load(file, *args, **kwargs)
            return _upgrade_sklearn_forest(obj)

        pickle.load = _pickle_load
    try:
        from sklearn.ensemble._forest import BaseForest
    except ImportError:
        BaseForest = None  # type: ignore[assignment]
    if (
        BaseForest is not None
        and _original_base_forest_tags is None
        and hasattr(BaseForest, "__sklearn_tags__")
    ):
        _original_base_forest_tags = BaseForest.__sklearn_tags__

        def _base_forest_tags(self):
            _upgrade_sklearn_forest(self)
            return _original_base_forest_tags(self)

        BaseForest.__sklearn_tags__ = _base_forest_tags
    _install_tree_predict_proba_normalization()
    _tree_pickle_compat_installed = True
    return True


def _uninstall_sklearn_legacy_tree_pickle_compat() -> None:
    global _tree_pickle_compat_installed
    global _original_check_node_ndarray, _original_pickle_load
    global _original_base_forest_tags, _original_dt_predict_proba
    if not _tree_pickle_compat_installed:
        return
    try:
        import pickle

        import sklearn.tree._tree as tree_mod
    except ImportError:
        _tree_pickle_compat_installed = False
        _original_check_node_ndarray = None
        _original_pickle_load = None
        _original_base_forest_tags = None
        _original_dt_predict_proba = None
        return
    if _original_check_node_ndarray is not None:
        tree_mod._check_node_ndarray = _original_check_node_ndarray
    if _original_pickle_load is not None:
        pickle.load = _original_pickle_load
    if _original_base_forest_tags is not None:
        try:
            from sklearn.ensemble._forest import BaseForest
        except ImportError:
            pass
        else:
            BaseForest.__sklearn_tags__ = _original_base_forest_tags
    if _original_dt_predict_proba is not None:
        try:
            from sklearn.tree import DecisionTreeClassifier
        except ImportError:
            pass
        else:
            DecisionTreeClassifier.predict_proba = _original_dt_predict_proba
    _original_check_node_ndarray = None
    _original_pickle_load = None
    _original_base_forest_tags = None
    _original_dt_predict_proba = None
    _tree_pickle_compat_installed = False


@contextmanager
def _sklearn_legacy_tree_pickle_compat() -> Iterator[None]:
    """Let TDC RandomForest pickles load on sklearn >= 1.3."""
    already = _tree_pickle_compat_installed
    install_sklearn_legacy_tree_pickle_compat()
    try:
        yield
    finally:
        if not already:
            _uninstall_sklearn_legacy_tree_pickle_compat()


def load_tdc_oracle(name: str) -> OracleFn:
    """Return a TDC oracle that scores a list of **valid** canonical SMILES.

    Invalid strings must be filtered before calling this function; TDC's
    fingerprint classifiers are not defined on unparseable input.
    """
    name = normalize_tdc_oracle_name(name)
    try:
        from tdc import Oracle
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "PyTDC is required for molopt property oracles. "
            "Install with: pip install PyTDC --no-deps\n"
            "Do not use plain `pip install PyTDC`: it pins transformers<4.51, "
            "which cannot load Qwen3-Embedding (needs >=4.51). This repo "
            "pins transformers==4.57.6; restore that if it was downgraded."
        ) from e
    prepare_tdc_oracle_dir()
    install_sklearn_legacy_tree_pickle_compat()
    with _tdc_mkdir_exist_ok():
        oracle = Oracle(name=name)
    _assert_tdc_oracle_alive(name, oracle)

    def _score(smiles: Sequence[str]):
        if not smiles:
            return []
        # TDC may return a scalar, a list, or an array; coerce_oracle_scores
        # normalizes that. Do not wrap with list() here — list(0.8) TypeError.
        return oracle(list(smiles))

    return _score


def load_property_oracle(
    name: str,
    *,
    factor_oracles: Mapping[str, OracleFn] | None = None,
) -> OracleFn:
    """TDC fingerprint score, or the GSK3B×JNK3 product for dual-kinase search."""
    key = normalize_property_oracle_name(name)
    factors = property_oracle_factors(key)
    if len(factors) == 1:
        if factor_oracles and factors[0] in factor_oracles:
            return factor_oracles[factors[0]]
        return load_tdc_oracle(factors[0])

    fns: list[OracleFn] = []
    for factor in factors:
        if factor_oracles and factor in factor_oracles:
            fns.append(factor_oracles[factor])
        else:
            fns.append(load_tdc_oracle(factor))

    def _score(smiles: Sequence[str]):
        texts = list(smiles)
        if not texts:
            return []
        parts = [
            coerce_oracle_scores(fn(texts), n=len(texts)) for fn in fns
        ]
        return [float(math.prod(vals)) for vals in zip(*parts, strict=True)]

    return _score


def _assert_tdc_oracle_alive(name: str, oracle) -> None:
    """Score TDC's published probe so a silent 0.0 RF failure cannot ship."""
    minimum = TDC_PROBE_MIN_SCORE.get(name)
    if minimum is None:
        return
    scores = coerce_oracle_scores(oracle([TDC_SKLEARN_PROBE_SMILES]), n=1)
    score = scores[0]
    if score < minimum:
        raise RuntimeError(
            f"{name} scored {score} on the TDC docs probe "
            f"(expected >= {minimum}). The RF pickle likely failed to unpickle "
            "and TDC returned 0.0. GSK3B loads its pickle on first score, not "
            "at Oracle() construction."
        )
    if score > 1.0 + 1e-6:
        raise RuntimeError(
            f"{name} scored {score} on the TDC docs probe (expected a "
            "probability in [0, 1]). sklearn likely returned leaf class "
            "counts instead of fractions; the predict_proba shim is missing."
        )
    print(f"[oracle] {name} probe={score:.4f} (min {minimum})", flush=True)


def _scalar_score(value: object, *, invalid_score: float) -> float:
    try:
        score = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(invalid_score)
    return float(invalid_score) if not math.isfinite(score) else score


def coerce_oracle_scores(
    raw: object,
    n: int,
    *,
    invalid_score: float = INVALID_SCORE,
) -> list[float]:
    """Normalize an oracle return value to ``n`` finite floats.

    Accepts a scalar, a sequence, or a 0-d array. Non-finite entries become
    ``invalid_score``. Raises ``ValueError`` if the length does not match ``n``.
    """
    if n < 0:
        raise ValueError("n must be nonnegative")
    if isinstance(raw, (str, bytes)):
        raise ValueError("oracle returned a string")
    shape = getattr(raw, "shape", None)
    if isinstance(raw, (int, float)) or shape == ():
        values = [_scalar_score(raw, invalid_score=invalid_score)]
    else:
        try:
            seq = list(raw)  # type: ignore[arg-type]
        except TypeError as e:
            raise ValueError("oracle returned a non-iterable score") from e
        values = [_scalar_score(item, invalid_score=invalid_score) for item in seq]
    if len(values) != n:
        raise ValueError(f"oracle returned {len(values)} scores for {n} molecules")
    return values


def _oracle_scores(
    oracle: SupportsOracle,
    smiles: Sequence[str],
    *,
    invalid_score: float,
) -> list[float]:
    """Call ``oracle`` on a valid-SMILES batch; fall back per molecule on failure."""
    if not smiles:
        return []
    try:
        return coerce_oracle_scores(
            oracle(smiles), n=len(smiles), invalid_score=invalid_score
        )
    except Exception:
        if len(smiles) == 1:
            raise
        return [
            _oracle_scores(oracle, [text], invalid_score=invalid_score)[0]
            for text in smiles
        ]


def score_molecules(
    smiles: Sequence[str],
    oracle: SupportsOracle,
    *,
    invalid_score: float = INVALID_SCORE,
    include_qed: bool = True,
) -> list[MoleculeScore]:
    """Score each SMILES; invalid molecules get ``invalid_score`` and skip the oracle."""
    texts = [str(s) for s in smiles]
    valid_idx: list[int] = []
    canonicals: list[str | None] = []
    for i, text in enumerate(texts):
        can = canonical_smiles(text)
        canonicals.append(can)
        if can is not None:
            valid_idx.append(i)
    scored_valid = (
        _oracle_scores(
            oracle,
            [canonicals[i] or texts[i] for i in valid_idx],
            invalid_score=invalid_score,
        )
        if valid_idx
        else []
    )
    by_index = dict(zip(valid_idx, scored_valid))
    rows: list[MoleculeScore] = []
    for i, text in enumerate(texts):
        can = canonicals[i]
        valid = can is not None
        rows.append(
            MoleculeScore(
                smiles=text,
                canonical=can,
                valid=valid,
                score=float(by_index[i]) if valid else float(invalid_score),
                qed=qed_value(text) if include_qed else None,
            )
        )
    return rows


def summarize_scores(
    rows: Sequence[MoleculeScore],
    *,
    thresholds: Sequence[float] = SCORE_THRESHOLDS,
) -> dict[str, float | int | None]:
    """Scalar summary of one split × one oracle.

    ``*_all`` includes invalid molecules at ``INVALID_SCORE``. ``*_valid`` is
    restricted to parseable molecules so a flat Sobol histogram is not
    confused with a dead generator.
    """
    n = len(rows)
    n_valid = sum(1 for row in rows if row.valid)
    valid_keys = {
        row.canonical for row in rows if row.valid and row.canonical is not None
    }
    all_scores = [row.score for row in rows]
    valid_scores = [row.score for row in rows if row.valid]
    qed_valid = [row.qed for row in rows if row.valid and row.qed is not None]
    out: dict[str, float | int | None] = {
        "n": n,
        "n_valid": n_valid,
        "n_unique_valid": len(valid_keys),
        "validity": (n_valid / n) if n else 0.0,
        "max_all": _max_or_none(all_scores),
        "mean_all": _mean_or_none(all_scores),
        "median_all": _median_or_none(all_scores),
        "p05_all": _quantile(all_scores, 0.05),
        "p95_all": _quantile(all_scores, 0.95),
        "max_valid": _max_or_none(valid_scores),
        "mean_valid": _mean_or_none(valid_scores),
        "median_valid": _median_or_none(valid_scores),
        "p05_valid": _quantile(valid_scores, 0.05),
        "p25_valid": _quantile(valid_scores, 0.25),
        "p75_valid": _quantile(valid_scores, 0.75),
        "p95_valid": _quantile(valid_scores, 0.95),
        "mean_qed_valid": _mean_or_none(qed_valid),
    }
    for tau in thresholds:
        key = _frac_key(tau)
        out[f"{key}_all"] = _frac_ge(all_scores, tau)
        out[f"{key}_valid"] = _frac_ge(valid_scores, tau)
    return out


def _frac_key(tau: float) -> str:
    text = f"{tau:g}".replace(".", "p")
    return f"frac_ge_{text}"


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _max_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(max(values))


def _median_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {q}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    weight = pos - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def _frac_ge(values: Sequence[float], tau: float) -> float | None:
    if not values:
        return None
    return float(sum(1 for v in values if v >= tau) / len(values))


def oracle_score_lists(rows: Sequence[MoleculeScore]) -> dict[str, list[float]]:
    """Score arrays for histograms: all observations, and valid molecules only."""
    return {
        "all": [float(row.score) for row in rows],
        "valid": [float(row.score) for row in rows if row.valid],
    }


def load_oracles(names: Sequence[str] | None = None) -> Mapping[str, OracleFn]:
    resolved = tuple(names) if names else TDC_ORACLE_NAMES
    return {name: load_property_oracle(name) for name in resolved}
