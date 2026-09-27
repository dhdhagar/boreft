"""Display helpers for truncating long text in logs and plots."""

from __future__ import annotations

# Stable column widths for training-time eval log tables. Tasks with long targets
# (molopt SMILES) widen the target/generated columns up to EVAL_LOG_WIDTH_MAX.
EVAL_LOG_TARGET_W = 22
EVAL_LOG_GENERATED_W = 40
EVAL_LOG_SIM_W = 6
EVAL_LOG_WIDTH_MAX = 56


def ellipsis_middle(
    text: str,
    *,
    start: int = 20,
    end: int = 20,
    max_total: int = 60,
) -> str:
    """Truncate long text: ``start...(+N)...end``."""
    text = " ".join(text.split())
    if len(text) <= max_total:
        return text
    if start + end >= len(text):
        return text[: max_total - 3] + "..."
    omitted = len(text) - start - end
    return f"{text[:start]}...(+{omitted})...{text[-end:]}"


def ellipsis_middle_fit(text: str, width: int) -> str:
    """``ellipsis_middle`` scaled to a fixed display column width."""
    if width <= 6:
        return text[:width]
    start = end = max(3, width // 3)
    return ellipsis_middle(text, start=start, end=end, max_total=width)


PLOT_LABEL_MAX = 24
PLOT_TITLE_LABEL_MAX = 32


def plot_label(text: str, max_chars: int = PLOT_LABEL_MAX) -> str:
    """Shorten a plot annotation so long targets (molopt SMILES) stay readable."""
    return ellipsis_middle_fit(text, max_chars)


def print_log_table(
    headers: tuple[str, ...],
    rows: list[tuple],
    *,
    widths: tuple[int, ...],
    align: tuple[str, ...] = ("<",),
    indent: str = "  ",
    sep: str = "  ",
) -> None:
    """Print a fixed-width table with a dashed separator under the header."""
    if len(widths) != len(headers):
        raise ValueError("widths must match headers")
    aligns = align if len(align) == len(headers) else align * len(headers)

    def fmt(cell: object, width: int, alignment: str) -> str:
        text = str(cell)
        if len(text) > width:
            text = text[: max(0, width - 3)] + "..."
        if alignment == ">":
            return f"{text:>{width}}"
        return f"{text:<{width}}"

    header_cells = [
        fmt(h, w, a) for h, w, a in zip(headers, widths, aligns)
    ]
    print(indent + sep.join(header_cells))
    print(indent + sep.join("-" * w for w in widths))
    for row in rows:
        if len(row) != len(headers):
            raise ValueError("row width must match headers")
        row_cells = [
            fmt(c, w, a) for c, w, a in zip(row, widths, aligns)
        ]
        print(indent + sep.join(row_cells))
