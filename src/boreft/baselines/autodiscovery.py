"""AutoDiscovery-style Monte Carlo tree search over generated solutions.

Reference: Agarwal et al., "AutoDiscovery: Open-ended Scientific Discovery via
Bayesian Surprise" (NeurIPS 2025)
https://arxiv.org/abs/2507.00310
https://github.com/allenai/autodiscovery-neurips

The task verifier's scalar objective replaces the paper's Bayesian-surprise
reward. Selection, expansion, and backpropagation follow the official repo's
``ucb1_recursive`` path: recursive UCB1, an OPRO-style scored-history prompt
over the selected root-to-node branch, and subtree visit/value updates. The
prompt itself matches OPRO; only the history subset changes with the branch.
Each expansion samples ``k_experiments`` one-solution completions (repo
``--k_experiments``, default 8), scores one at random, and keeps the rest as
untried solutions on that node. A later visit to the same node draws from that
pool before generating again (repo ``get_next_experiment``).

Launch via the shared baseline CLI, for example::

    python -m boreft.baselines.search \
      --baseline autodiscovery \
      --task semantle \
      --task-description "Generate an English word as a guess to find the hidden word (only the word, without any decoration or formatting)." \
      --target computer \
      --reft-output-dir outputs/1784053292 \
      --search-dir outputs/1784053292/search/autodiscovery-computer \
      --budget 500 \
      --warmstart-count 10 \
      --warmstart-source checkpoint \
      --batch-size 1 \
      --seeds 1 2 3 \
      --overwrite
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Mapping, Sequence

from .base import (
    BaselineObservation,
    Candidate,
    JSONValue,
    SearchContext,
    solution_key,
)
from .opro import _build_prompt, _parse_solution

_BLANK_SAMPLE_RETRIES = 3
_ROOT_ID = 0


@dataclass(frozen=True)
class AutoDiscoveryConfig:
    """MCTS knobs matching the AutoDiscovery NeurIPS implementation.

    ``exploration_constant`` is ``C`` in UCT (repo ``--exploration_weight``,
    default 2.0). Selection follows repo ``ucb1_recursive``; ``max_depth`` is
    the only expandability cap. ``k_experiments`` is ``--k_experiments`` (pool
    of one-solution completions using the OPRO prompt). ``parent_context`` is
    ``--k_parents`` (path nodes included as that prompt's history; ``None``
    keeps the full branch).
    """

    exploration_constant: float = 2.0
    k_experiments: int = 8
    max_depth: int | None = None
    parent_context: int | None = 3

    def __post_init__(self) -> None:
        if self.exploration_constant < 0:
            raise ValueError("exploration_constant must be nonnegative")
        if self.k_experiments < 1:
            raise ValueError("k_experiments must be positive")
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError("max_depth must be positive or None")
        if self.parent_context is not None and self.parent_context < 1:
            raise ValueError("parent_context must be positive or None")


@dataclass
class MCTSNode:
    """One scored solution, plus a dummy root at ``node_id=0``."""

    node_id: int
    parent_id: int | None
    depth: int
    solution: str | None = None
    observation_index: int | None = None
    score: float | None = None
    visits: int = 0
    value: float = 0.0
    children_ids: list[int] = field(default_factory=list)
    untried_solutions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "solution": self.solution,
            "observation_index": self.observation_index,
            "score": self.score,
            "visits": self.visits,
            "value": self.value,
            "children_ids": list(self.children_ids),
            "untried_solutions": list(self.untried_solutions),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "MCTSNode":
        children = data.get("children_ids", [])
        if not isinstance(children, list):
            raise ValueError("MCTS node children_ids must be a list")
        parent_id = data.get("parent_id")
        observation_index = data.get("observation_index")
        score = data.get("score")
        solution = data.get("solution")
        untried = data.get("untried_solutions", [])
        if not isinstance(untried, list):
            raise ValueError("MCTS node untried_solutions must be a list")
        return cls(
            node_id=int(data["node_id"]),
            parent_id=None if parent_id is None else int(parent_id),
            depth=int(data["depth"]),
            solution=None if solution is None else str(solution),
            observation_index=(
                None if observation_index is None else int(observation_index)
            ),
            score=None if score is None else float(score),
            visits=int(data.get("visits", 0)),
            value=float(data.get("value", 0.0)),
            children_ids=[int(child_id) for child_id in children],
            untried_solutions=[str(item) for item in untried],
        )


def ucb1(
    node: MCTSNode,
    parent: MCTSNode | None,
    exploration_constant: float,
) -> float:
    """UCT from AutoDiscovery Eq. (6) / repo ``ucb1``.

    Unvisited nodes score ``+inf``. Exploitation is mean subtree reward
    ``value / visits``; exploration is ``C * sqrt(2 log N_parent / N)``.
    """
    if node.visits <= 0:
        return math.inf
    exploitation = node.value / node.visits
    if parent is None or parent.visits <= 0:
        return exploitation
    exploration = math.sqrt(2.0 * math.log(parent.visits) / node.visits)
    return exploitation + exploration_constant * exploration


def can_expand(node: MCTSNode, config: AutoDiscoveryConfig) -> bool:
    """Whether ``node`` may receive another child (``max_depth`` only)."""
    return config.max_depth is None or node.depth < config.max_depth


def pick_untried(node: MCTSNode, rng: random.Random) -> str:
    """Pop one uniformly random untried solution (repo ``get_next_experiment``)."""
    if not node.untried_solutions:
        raise ValueError("MCTS node has no untried solutions")
    index = rng.randrange(len(node.untried_solutions))
    return node.untried_solutions.pop(index)


def _dump_rng(rng: random.Random | None) -> JSONValue:
    if rng is None:
        return None
    version, internals, gauss = rng.getstate()
    return {
        "version": int(version),
        "internals": [int(value) for value in internals],
        "gauss": None if gauss is None else float(gauss),
    }


def _load_rng(raw: object) -> random.Random | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("autodiscovery rng state must be an object")
    internals = raw.get("internals", [])
    if not isinstance(internals, list):
        raise ValueError("autodiscovery rng internals must be a list")
    rng = random.Random()
    rng.setstate(
        (int(raw["version"]), tuple(int(value) for value in internals), raw.get("gauss"))
    )
    return rng


def _parent(nodes: Mapping[int, MCTSNode], node: MCTSNode) -> MCTSNode | None:
    if node.parent_id is None:
        return None
    return nodes.get(node.parent_id)


def _selection_key(score: float, node_id: int) -> tuple[int, float, int]:
    """Highest UCB first; ``+inf`` before finite values; lower id on ties."""
    if score == math.inf:
        return (0, 0.0, node_id)
    if score == -math.inf:
        return (2, 0.0, node_id)
    return (1, -score, node_id)


def _ucb_key(
    nodes: Mapping[int, MCTSNode],
    node: MCTSNode,
    exploration_constant: float,
) -> tuple[int, float, int]:
    return _selection_key(
        ucb1(node, _parent(nodes, node), exploration_constant),
        node.node_id,
    )


def _best_by_ucb(
    nodes: Mapping[int, MCTSNode],
    candidates: Sequence[MCTSNode],
    exploration_constant: float,
) -> MCTSNode:
    return min(candidates, key=lambda item: _ucb_key(nodes, item, exploration_constant))


def select_expandable(
    nodes: Mapping[int, MCTSNode],
    config: AutoDiscoveryConfig,
    *,
    start_id: int = _ROOT_ID,
) -> MCTSNode:
    """Recursive UCB1 from the AutoDiscovery repo (``ucb1_recursive``).

    Rank the current node against its children by UCB1. Expand it when it wins;
    otherwise recurse into the best child. Nodes at ``max_depth`` are skipped.
    Ties break toward the lower ``node_id`` (creation order).
    """
    if start_id not in nodes:
        raise ValueError(f"MCTS node {start_id} is missing from the tree")
    constant = config.exploration_constant

    def choose(node: MCTSNode) -> MCTSNode:
        ranked = sorted(
            [node]
            + [
                nodes[child_id]
                for child_id in node.children_ids
                if child_id in nodes
            ],
            key=lambda item: _ucb_key(nodes, item, constant),
        )
        for best in ranked:
            if not can_expand(best, config):
                continue
            if best.node_id == node.node_id:
                return best
            return choose(best)
        return node

    chosen = choose(nodes[start_id])
    if can_expand(chosen, config):
        return chosen
    expandable = [item for item in nodes.values() if can_expand(item, config)]
    if not expandable:
        return chosen
    return _best_by_ucb(nodes, expandable, constant)


def path_from_root(
    nodes: Mapping[int, MCTSNode],
    node_id: int,
) -> list[MCTSNode]:
    """Scored nodes from the dummy root's child down to ``node_id``."""
    path: list[MCTSNode] = []
    current = nodes.get(node_id)
    seen: set[int] = set()
    while current is not None and current.node_id not in seen:
        seen.add(current.node_id)
        if current.solution is not None:
            path.append(current)
        if current.parent_id is None:
            break
        current = nodes.get(current.parent_id)
    path.reverse()
    return path


class _ScoredHistoryItem:
    """Duck-typed scored row for :func:`boreft.baselines.opro._build_prompt`."""

    __slots__ = ("index", "score", "solution")

    def __init__(self, index: int, solution: str, score: float) -> None:
        self.index = index
        self.solution = solution
        self.score = score


def _branch_history(path: Sequence[MCTSNode]) -> list[_ScoredHistoryItem]:
    """Branch nodes with solutions, ordered like OPRO (lowest score to highest)."""
    history = [
        _ScoredHistoryItem(
            index=(
                -1
                if node.observation_index is None
                else int(node.observation_index)
            ),
            solution=node.solution,
            score=0.0 if node.score is None else float(node.score),
        )
        for node in path
        if node.solution
    ]
    history.sort(key=lambda item: (item.score, item.index))
    return history


def build_branch_prompt(
    task_description: str,
    path: Sequence[MCTSNode],
    *,
    task: str | None = None,
) -> str:
    """OPRO scored-history prompt over the selected branch.

    The wording matches :func:`boreft.baselines.opro._build_prompt`. Only the
    history set differs: this is the MCTS path, not OPRO's global top-k.
    """
    return _build_prompt(task_description, _branch_history(path), task=task)


def _sample_solutions(
    context: SearchContext,
    prompt: str,
    count: int,
) -> list[str]:
    for _ in range(_BLANK_SAMPLE_RETRIES):
        texts = list(context.generate(prompt, count, context.generation_options))
        sampled: list[str] = []
        seen: set[str] = set()
        for text in texts:
            solution = _parse_solution(text)
            if solution is None:
                continue
            key = solution_key(solution)
            if key in seen:
                continue
            seen.add(key)
            sampled.append(solution)
            if len(sampled) >= count:
                break
        if sampled:
            return sampled
    return []


def _link_children(nodes: Mapping[int, MCTSNode]) -> None:
    """Rebuild child lists from ``parent_id`` in creation order."""
    for node in nodes.values():
        node.children_ids = []
    for node_id in sorted(nodes):
        node = nodes[node_id]
        parent = _parent(nodes, node)
        if parent is not None and node.node_id not in parent.children_ids:
            parent.children_ids.append(node.node_id)


class AutoDiscoveryBaseline:
    name = "autodiscovery"

    def __init__(self, config: AutoDiscoveryConfig | None = None) -> None:
        self.config = config or AutoDiscoveryConfig()
        self._nodes: dict[int, MCTSNode] = {}
        self._next_id = 1
        self._observation_ids: dict[int, int] = {}
        self._rng: random.Random | None = None

    def propose(
        self,
        context: SearchContext,
        history: Sequence[BaselineObservation],
        count: int,
    ) -> Sequence[Candidate]:
        if count < 1:
            raise ValueError("candidate count must be positive")
        rng = self._ensure_rng(context.seed)
        self._ensure_tree(history)
        parent = select_expandable(self._nodes, self.config)
        path = path_from_root(self._nodes, parent.node_id)
        if self.config.parent_context is not None:
            path = path[-self.config.parent_context :]
        prompt = build_branch_prompt(
            context.task_description, path, task=context.task
        )
        history_indices = [
            item.index for item in _branch_history(path) if item.index >= 0
        ]
        reused_untried = bool(parent.untried_solutions)
        if not parent.untried_solutions:
            sampled = _sample_solutions(
                context, prompt, self.config.k_experiments
            )
            if not sampled:
                raise ValueError(
                    f"{self.name} received only blank samples from the base model"
                )
            parent.untried_solutions.extend(sampled)
        solution = pick_untried(parent, rng)
        node_id = self._next_id
        self._next_id += 1
        key = solution_key(solution)
        seen_history = {solution_key(item.solution) for item in history}
        return [
            Candidate(
                solution=solution,
                metadata={
                    "prompt": prompt,
                    "node_id": node_id,
                    "parent_id": parent.node_id,
                    "depth": parent.depth + 1,
                    "history_indices": list(history_indices),
                    "is_repeat_proposal": key in seen_history,
                    "reused_untried": reused_untried,
                },
            )
        ]

    def observe(self, observations: Sequence[BaselineObservation]) -> None:
        for item in observations:
            self._attach_observation(item)

    def state_dict(self) -> dict:
        return {
            "next_id": self._next_id,
            "nodes": [self._nodes[node_id].to_dict() for node_id in sorted(self._nodes)],
            "rng": _dump_rng(self._rng),
        }

    def load_state_dict(self, state: Mapping) -> None:
        if not state:
            self._reset_tree()
            return
        nodes_raw = state.get("nodes", [])
        if not isinstance(nodes_raw, list):
            raise ValueError("autodiscovery method state nodes must be a list")
        nodes: dict[int, MCTSNode] = {}
        observation_ids: dict[int, int] = {}
        for item in nodes_raw:
            if not isinstance(item, Mapping):
                raise ValueError("autodiscovery method state node must be an object")
            node = MCTSNode.from_dict(item)
            nodes[node.node_id] = node
            if node.observation_index is not None:
                observation_ids[node.observation_index] = node.node_id
        if nodes and _ROOT_ID not in nodes:
            raise ValueError("autodiscovery method state is missing the root node")
        _link_children(nodes)
        self._nodes = nodes
        self._observation_ids = observation_ids
        max_existing = max(nodes, default=_ROOT_ID)
        self._next_id = max(int(state.get("next_id", 1)), max_existing + 1)
        self._rng = _load_rng(state.get("rng"))

    def _reset_tree(self) -> None:
        self._nodes = {}
        self._next_id = 1
        self._observation_ids = {}
        self._rng = None

    def _ensure_rng(self, seed: int) -> random.Random:
        if self._rng is None:
            self._rng = random.Random(seed)
        return self._rng

    def _ensure_tree(self, history: Sequence[BaselineObservation]) -> None:
        if not self._nodes:
            self._nodes = {
                _ROOT_ID: MCTSNode(node_id=_ROOT_ID, parent_id=None, depth=0)
            }
            self._next_id = 1
            self._observation_ids = {}
        for item in history:
            self._attach_observation(item)

    def _attach_observation(self, item: BaselineObservation) -> int:
        existing = self._observation_ids.get(item.index)
        if existing is not None:
            return existing
        metadata = item.candidate_metadata
        raw_node_id = metadata.get("node_id")
        node_id = self._next_id if raw_node_id is None else int(raw_node_id)
        node = self._nodes.get(node_id)
        if node is not None and node.observation_index is not None:
            self._observation_ids[item.index] = node_id
            return node_id
        raw_parent_id = metadata.get("parent_id")
        parent_id = _ROOT_ID if raw_parent_id is None else int(raw_parent_id)
        if parent_id not in self._nodes:
            parent_id = _ROOT_ID
        parent = self._nodes[parent_id]
        depth_meta = metadata.get("depth")
        depth = parent.depth + 1 if depth_meta is None else int(depth_meta)
        if node is None:
            node = MCTSNode(
                node_id=node_id,
                parent_id=parent_id,
                depth=depth,
                solution=item.solution,
            )
            self._nodes[node_id] = node
            if node_id not in parent.children_ids:
                parent.children_ids.append(node_id)
        node.solution = item.solution
        node.observation_index = item.index
        node.score = item.score
        if node.visits == 0:
            self._backpropagate(node_id, item.score)
        self._observation_ids[item.index] = node_id
        self._next_id = max(self._next_id, node_id + 1)
        return node_id

    def _backpropagate(self, node_id: int, reward: float) -> None:
        """Add one visit and ``reward`` along the path to the dummy root."""
        current = self._nodes.get(node_id)
        seen: set[int] = set()
        while current is not None and current.node_id not in seen:
            seen.add(current.node_id)
            current.visits += 1
            current.value += reward
            if current.parent_id is None:
                break
            current = self._nodes.get(current.parent_id)
