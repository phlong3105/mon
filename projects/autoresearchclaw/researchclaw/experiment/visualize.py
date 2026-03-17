"""Experiment result visualization.

Generates charts from experiment run data:
- Metric improvement trajectory (line chart)
- Multi-experiment comparison (bar chart)
- Pipeline execution timeline (horizontal bar)
- Iteration score history (line chart)

All functions return Path to the generated image, or None if matplotlib
is unavailable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Metrics to exclude from comparison charts (timing, meta, non-scientific)
_EXCLUDED_METRICS: set[str] = {
    "time_budget_sec", "elapsed_sec", "elapsed_time", "execution_time",
    "wall_time", "runtime_sec", "total_time", "timeout",
    "seed", "seed_count", "n_seeds", "num_seeds",
    "success_rate", "num_conditions", "total_conditions",
    "calibration_iterations",
}

# Prefixes that indicate meta/timing metrics
_EXCLUDED_PREFIXES: tuple[str, ...] = ("time_", "runtime_", "elapsed_", "wall_")


def _is_excluded_metric(name: str) -> bool:
    """Return True if *name* is a timing/meta metric that shouldn't be charted."""
    low = name.lower()
    if low in _EXCLUDED_METRICS:
        return True
    return any(low.startswith(p) for p in _EXCLUDED_PREFIXES)


def _shorten_label(name: str, max_len: int = 25) -> str:
    """Shorten a metric label for chart readability."""
    if len(name) <= max_len:
        return name
    return name[: max_len - 1] + "\u2026"


def _ensure_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 1. Metric improvement trajectory
# ---------------------------------------------------------------------------


def plot_metric_trajectory(
    runs: list[dict[str, Any]],
    metric_key: str,
    output_path: Path,
    *,
    title: str = "",
) -> Path | None:
    """Plot metric values across runs as a line chart.

    *runs* should be a list of dicts with a ``metrics`` field containing
    the metric keyed by *metric_key*.
    """
    if not HAS_MATPLOTLIB or not runs:
        return None

    values: list[float] = []
    labels: list[str] = []
    for i, r in enumerate(runs):
        m = r.get("metrics") or r.get("key_metrics") or {}
        if isinstance(m, dict) and metric_key in m:
            try:
                values.append(float(m[metric_key]))
                labels.append(r.get("run_id", f"run-{i + 1}"))
            except (ValueError, TypeError):
                continue

    if not values:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        range(len(values)), values, "o-", color="#2196F3", linewidth=2, markersize=6
    )
    ax.set_xlabel("Run")
    ax.set_ylabel(metric_key)
    ax.set_title(title or f"{metric_key} Trajectory")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(_ensure_dir(output_path), dpi=150)
    plt.close(fig)
    logger.info("Saved metric trajectory: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 2. Multi-experiment comparison (bar chart)
# ---------------------------------------------------------------------------


def plot_experiment_comparison(
    metrics_summary: dict[str, dict[str, float]],
    output_path: Path,
    *,
    title: str = "Experiment Results Comparison",
) -> Path | None:
    """Bar chart comparing mean values across all metrics."""
    if not HAS_MATPLOTLIB or not metrics_summary:
        return None

    # Filter out timing/meta metrics that distort chart scale
    filtered = {k: v for k, v in metrics_summary.items() if not _is_excluded_metric(k)}
    if not filtered:
        return None

    names = list(filtered.keys())
    means = [filtered[n].get("mean", 0) for n in names]
    mins = [filtered[n].get("min", 0) for n in names]
    maxs = [filtered[n].get("max", 0) for n in names]

    fig, ax = plt.subplots(figsize=(min(18, max(6, len(names) * 1.5)), 4))
    x = range(len(names))
    bars = ax.bar(x, means, color="#4CAF50", alpha=0.8, label="Mean")
    # Error bars showing min-max range
    for i, (lo, hi, mean) in enumerate(zip(mins, maxs, means)):
        ax.plot([i, i], [lo, hi], color="#333", linewidth=1.5)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.set_xticks(list(x))
    ax.set_xticklabels([_shorten_label(n) for n in names], rotation=30, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(_ensure_dir(output_path), dpi=150)
    plt.close(fig)
    logger.info("Saved experiment comparison: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 2b. Condition comparison (aggregated per-condition bar chart)
# ---------------------------------------------------------------------------


def plot_condition_comparison(
    condition_summaries: dict[str, dict[str, Any]],
    output_path: Path,
    *,
    metric_key: str = "primary_metric",
    title: str = "Method Comparison",
) -> Path | None:
    """Bar chart comparing conditions with mean +/- 95% CI error bars.

    *condition_summaries* maps condition name → dict with ``metrics``,
    ``ci95_low``, ``ci95_high``, ``n_seeds``.
    """
    if not HAS_MATPLOTLIB or not condition_summaries:
        return None

    names: list[str] = []
    means: list[float] = []
    ci_low: list[float] = []
    ci_high: list[float] = []

    for cond, info in condition_summaries.items():
        m = info.get("metrics", {})
        # Prefer aggregated mean over single-run value
        mean_val = m.get(f"{metric_key}_mean") or m.get(metric_key)
        if mean_val is None:
            continue
        fmean = float(mean_val)
        names.append(cond.replace("_", " ").title())
        means.append(fmean)
        ci_low.append(float(info.get("ci95_low", fmean)))
        ci_high.append(float(info.get("ci95_high", fmean)))

    if not names:
        return None

    yerr_lo = [max(0, m - lo) for m, lo in zip(means, ci_low)]
    yerr_hi = [max(0, hi - m) for m, hi in zip(means, ci_high)]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.8), 5))
    x = range(len(names))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#607D8B"]
    bar_colors = [colors[i % len(colors)] for i in range(len(names))]
    ax.bar(x, means, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.errorbar(
        x, means, yerr=[yerr_lo, yerr_hi],
        fmt="none", ecolor="#333", capsize=4, capthick=1.5, linewidth=1.5,
    )

    # Add value labels
    _offset = max(yerr_hi) * 0.1 if yerr_hi and max(yerr_hi) > 0 else max(means) * 0.02
    for i, m in enumerate(means):
        ax.text(i, m + _offset, f"{m:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xlabel("Method / Condition", fontsize=11)
    ax.set_ylabel(metric_key.replace("_", " ").title(), fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(_ensure_dir(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved condition comparison: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 3. Pipeline execution timeline
# ---------------------------------------------------------------------------


def plot_pipeline_timeline(
    stage_results: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str = "Pipeline Execution Timeline",
) -> Path | None:
    """Horizontal bar chart showing execution time per stage."""
    if not HAS_MATPLOTLIB or not stage_results:
        return None

    labels: list[str] = []
    durations: list[float] = []
    colors: list[str] = []

    for r in stage_results:
        name = r.get("stage_name", r.get("stage", "?"))
        elapsed = r.get("elapsed_sec", 0)
        status = r.get("status", "done")
        labels.append(str(name))
        durations.append(float(elapsed) if elapsed else 1.0)
        colors.append("#4CAF50" if status == "done" else "#F44336")

    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.3)))
    y = range(len(labels))
    ax.barh(list(y), durations, color=colors, alpha=0.8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Time (seconds)")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(_ensure_dir(output_path), dpi=150)
    plt.close(fig)
    logger.info("Saved pipeline timeline: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 4. Iteration score history
# ---------------------------------------------------------------------------


def plot_iteration_scores(
    scores: list[float | None],
    output_path: Path,
    *,
    threshold: float = 7.0,
    title: str = "Quality Score by Iteration",
) -> Path | None:
    """Line chart of quality scores across iterations."""
    if not HAS_MATPLOTLIB or not scores:
        return None

    valid = [(i + 1, s) for i, s in enumerate(scores) if s is not None]
    if not valid:
        return None

    iters, vals = zip(*valid)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(iters, vals, "o-", color="#9C27B0", linewidth=2, markersize=8)
    ax.axhline(
        y=threshold,
        color="#F44336",
        linestyle="--",
        alpha=0.7,
        label=f"Threshold ({threshold})",
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Quality Score")
    ax.set_title(title)
    ax.set_ylim(0, 10.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(_ensure_dir(output_path), dpi=150)
    plt.close(fig)
    logger.info("Saved iteration scores: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 5. All-in-one: generate all charts from run directory
# ---------------------------------------------------------------------------


def generate_all_charts(
    run_dir: Path,
    output_dir: Path | None = None,
    *,
    metric_key: str = "val_loss",
) -> list[Path]:
    """Scan run_dir and generate all applicable charts.

    Returns list of generated image paths.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available — skipping chart generation")
        return []

    if output_dir is None:
        output_dir = run_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []

    # Collect experiment runs
    runs: list[dict[str, Any]] = []
    for stage_subdir in sorted(run_dir.glob("stage-*/runs")):
        for run_file in sorted(stage_subdir.glob("*.json")):
            try:
                data = json.loads(run_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    runs.append(data)
            except (json.JSONDecodeError, OSError):
                continue

    # 1. Metric trajectory
    path = plot_metric_trajectory(
        runs, metric_key, output_dir / "metric_trajectory.png"
    )
    if path:
        generated.append(path)

    # 2. Experiment comparison (raw metrics — capped at 18in width)
    summary_path = run_dir / "stage-14" / "experiment_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

            # 2a. Condition comparison (aggregated, preferred)
            cs = summary.get("condition_summaries", {})
            if cs:
                path = plot_condition_comparison(
                    cs,
                    output_dir / "method_comparison.png",
                    metric_key=metric_key,
                    title="Method Comparison (Mean ± 95% CI)",
                )
                if path:
                    generated.append(path)

            # 2b. Raw metrics comparison (fallback, limited to top 20)
            ms = summary.get("metrics_summary", {})
            if ms:
                # Filter out timing/meta metrics first
                ms = {k: v for k, v in ms.items() if not _is_excluded_metric(k)}
                # Limit to top 20 metrics by mean value to avoid giant charts
                if len(ms) > 20:
                    top = sorted(ms.items(),
                                 key=lambda kv: kv[1].get("mean", 0),
                                 reverse=True)[:20]
                    ms = dict(top)
                path = plot_experiment_comparison(
                    ms, output_dir / "experiment_comparison.png"
                )
                if path:
                    generated.append(path)
        except (json.JSONDecodeError, OSError):
            pass

    # 3. Iteration scores
    iter_path = run_dir / "iteration_summary.json"
    if iter_path.exists():
        try:
            iter_data = json.loads(iter_path.read_text(encoding="utf-8"))
            scores = iter_data.get("iteration_scores", [])
            threshold = iter_data.get("quality_threshold", 7.0)
            path = plot_iteration_scores(
                scores, output_dir / "iteration_scores.png", threshold=threshold
            )
            if path:
                generated.append(path)
        except (json.JSONDecodeError, OSError):
            pass

    logger.info("Generated %d chart(s) in %s", len(generated), output_dir)
    return generated
