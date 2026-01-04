"""Visualization utilities for model comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_metrics_comparison_chart(
    results: dict[str, dict[str, float]],
    metrics: list[str] | None = None,
    primary_metric: str = "pr_auc",
    output_dir: Path = Path("artifacts/plots"),
) -> Path:
    """Create a grouped bar chart comparing metrics across models."""
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]

    model_names = list(results.keys())
    n_models = len(model_names)
    n_metrics = len(metrics)

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n_models)
    width = 0.12
    offsets = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * width

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12", "#1abc9c"]

    for i, metric in enumerate(metrics):
        values = [results[m][metric] for m in model_names]
        bars = ax.bar(
            x + offsets[i],
            values,
            width,
            label=metric.capitalize(),
            color=colors[i],
            alpha=1.0 if metric == primary_metric else 0.7,
            edgecolor="black" if metric == primary_metric else "none",
            linewidth=2 if metric == primary_metric else 0,
        )

        for bar, val in zip(bars, values):
            ax.annotate(
                f"{val:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=45,
            )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        f"Model Comparison (Primary Metric: {primary_metric.upper()})", fontsize=14
    )
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in model_names])
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / "model_comparison.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Chart saved to {chart_path}")
    return chart_path


def create_metric_table(
    results: dict[str, dict[str, float]],
    primary_metric: str = "pr_auc",
) -> str:
    """Create a formatted ASCII table for console output."""
    # Find best model
    best_model = max(results.keys(), key=lambda m: results[m][primary_metric])

    # Header
    header = f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC_AUC':>10} {'PR_AUC':>10}"
    separator = "-" * len(header)

    lines = [separator, header, separator]

    for model_name, model_metrics in results.items():
        marker = " *" if model_name == best_model else "  "
        line = f"{model_name:<20}{marker} {model_metrics['accuracy']:>10.4f} {model_metrics['precision']:>10.4f} {model_metrics['recall']:>10.4f} {model_metrics['f1']:>10.4f} {model_metrics['roc_auc']:>10.4f} {model_metrics['pr_auc']:>10.4f}"
        lines.append(line)

    lines.append(separator)
    lines.append(f"* Best model based on {primary_metric}")

    return "\n".join(lines)


def create_pr_curve(
    model_predictions: dict[str, tuple],
    output_dir: Path = Path("artifacts/plots"),
) -> Path:
    """
    Create Precision-Recall curve for multiple models.

    Args:
        model_predictions: Dict of model_name -> (y_true, y_prob) tuples
        output_dir: Directory to save the plot

    Returns:
        Path to the saved chart
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12", "#1abc9c"]

    for i, (model_name, (y_true, y_prob)) in enumerate(model_predictions.items()):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)

        ax.plot(
            recall,
            precision,
            color=colors[i % len(colors)],
            linewidth=2,
            label=f"{model_name} (PR-AUC = {pr_auc:.3f})",
        )

    # Add baseline (random classifier)
    baseline = sum(model_predictions[list(model_predictions.keys())[0]][0]) / len(
        model_predictions[list(model_predictions.keys())[0]][0]
    )
    ax.axhline(y=baseline, color="gray", linestyle="--", label=f"Baseline ({baseline:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve Comparison", fontsize=14)
    ax.legend(loc="best")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / "pr_curve.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"PR curve saved to {chart_path}")
    return chart_path


def create_lift_gain_chart(
    model_predictions: dict[str, tuple],
    output_dir: Path = Path("artifacts/plots"),
) -> Path:
    """
    Create Lift and Gain charts for multiple models.

    Args:
        model_predictions: Dict of model_name -> (y_true, y_prob) tuples
        output_dir: Directory to save the plot

    Returns:
        Path to the saved chart
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12", "#1abc9c"]

    for i, (model_name, (y_true, y_prob)) in enumerate(model_predictions.items()):
        # Sort by predicted probability descending
        sorted_indices = np.argsort(y_prob)[::-1]
        y_true_sorted = np.array(y_true)[sorted_indices]

        # Calculate cumulative gains
        n_samples = len(y_true)
        n_positives = sum(y_true)
        percentiles = np.arange(1, n_samples + 1) / n_samples * 100
        cumulative_positives = np.cumsum(y_true_sorted)
        gain = cumulative_positives / n_positives * 100

        # Calculate lift
        response_rate = cumulative_positives / np.arange(1, n_samples + 1)
        baseline_rate = n_positives / n_samples
        lift = response_rate / baseline_rate

        # Gain chart
        axes[0].plot(
            percentiles,
            gain,
            color=colors[i % len(colors)],
            linewidth=2,
            label=model_name,
        )

        # Lift chart
        axes[1].plot(
            percentiles,
            lift,
            color=colors[i % len(colors)],
            linewidth=2,
            label=model_name,
        )

    # Add baseline to gain chart
    axes[0].plot([0, 100], [0, 100], "k--", linewidth=1, label="Random")
    axes[0].set_xlabel("% of Population", fontsize=12)
    axes[0].set_ylabel("% of Positive Cases Captured", fontsize=12)
    axes[0].set_title("Cumulative Gain Chart", fontsize=14)
    axes[0].legend(loc="lower right")
    axes[0].set_xlim([0, 100])
    axes[0].set_ylim([0, 100])
    axes[0].grid(alpha=0.3)

    # Add baseline to lift chart
    axes[1].axhline(y=1, color="k", linestyle="--", linewidth=1, label="Random")
    axes[1].set_xlabel("% of Population", fontsize=12)
    axes[1].set_ylabel("Lift", fontsize=12)
    axes[1].set_title("Lift Chart", fontsize=14)
    axes[1].legend(loc="upper right")
    axes[1].set_xlim([0, 100])
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / "lift_gain_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Lift/Gain chart saved to {chart_path}")
    return chart_path
