"""AST-level diff classifier for two reward-function source strings.

Used by `closed-loop-fitness` (and any future multi-iteration training)
to decide what to do with the replay buffer when the reward function
changes between iterations. Pure function, no I/O.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
from dataclasses import dataclass
from typing import Literal

_SIMILARITY_THRESHOLD = 0.7


RewardDiffKind = Literal["IDENTICAL", "NUMERIC_DIFF", "STRUCTURAL_DIFF", "TOTAL_REWRITE"]


@dataclass(frozen=True)
class RewardDiff:
    kind: RewardDiffKind
    similarity: float
    diff_summary: str


def _ast_signature(tree: ast.AST) -> list[str]:
    """Canonical structure-aware signature: hides numeric constant *values* but keeps shape."""
    tokens: list[str] = []
    for node in ast.walk(tree):
        tokens.append(type(node).__name__)
        if isinstance(node, ast.Name):
            tokens.append(node.id)
        elif isinstance(node, ast.Attribute):
            tokens.append(node.attr)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                tokens.append("<NUM>")
            else:
                tokens.append(repr(node.value))
    return tokens


def _changed_constants(old_tree: ast.AST, new_tree: ast.AST) -> list[tuple[object, object]]:
    """Pair up numeric constants in parse order; list pairs that differ."""
    old_nums = [
        n.value
        for n in ast.walk(old_tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, (int, float))
        and not isinstance(n.value, bool)
    ]
    new_nums = [
        n.value
        for n in ast.walk(new_tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, (int, float))
        and not isinstance(n.value, bool)
    ]
    return [(a, b) for a, b in zip(old_nums, new_nums) if a != b]


def diff_rewards(old_src: str, new_src: str) -> RewardDiff:
    """Classify two reward source strings into one of four kinds.

    The classification is conservative: when in doubt, prefer "more different".
    Unparseable input falls back to TOTAL_REWRITE rather than raising.
    """
    if old_src == new_src:
        return RewardDiff(kind="IDENTICAL", similarity=1.0, diff_summary="bytes-equal")

    if hashlib.sha256(old_src.encode("utf-8")).hexdigest() == hashlib.sha256(
        new_src.encode("utf-8")
    ).hexdigest():
        # Unreachable under normal Python, but guards against locale/encoding edge cases
        return RewardDiff(kind="IDENTICAL", similarity=1.0, diff_summary="sha256-equal")

    try:
        old_tree = ast.parse(old_src)
        new_tree = ast.parse(new_src)
    except SyntaxError as e:
        return RewardDiff(
            kind="TOTAL_REWRITE",
            similarity=0.0,
            diff_summary=f"unparseable input ({e.msg}); conservative fallback",
        )

    sig_old = _ast_signature(old_tree)
    sig_new = _ast_signature(new_tree)

    if sig_old == sig_new:
        changed = _changed_constants(old_tree, new_tree)
        summary = (
            f"{len(changed)} numeric constants changed: "
            + ", ".join(f"{a}->{b}" for a, b in changed[:5])
        )
        if len(changed) > 5:
            summary += f", ... ({len(changed) - 5} more)"
        return RewardDiff(kind="NUMERIC_DIFF", similarity=1.0, diff_summary=summary)

    ratio = difflib.SequenceMatcher(None, sig_old, sig_new).ratio()
    summary = (
        f"signature lengths old={len(sig_old)} new={len(sig_new)}, "
        f"SequenceMatcher.ratio={ratio:.3f}"
    )
    if ratio > _SIMILARITY_THRESHOLD:
        return RewardDiff(kind="STRUCTURAL_DIFF", similarity=ratio, diff_summary=summary)
    return RewardDiff(kind="TOTAL_REWRITE", similarity=ratio, diff_summary=summary)
