"""Model comparison and selection utilities."""

from src.utils.logger import get_logger

logger = get_logger(__name__)


def select_best_model(
    results: dict[str, dict],
    primary_metric: str = "recall",
    metric_direction: str = "maximize",
) -> tuple[str, dict]:
    """
    Select the best model based on primary metric.

    Args:
        results: Dictionary mapping model_name -> {"metrics": {...}, "run_id": ...}
        primary_metric: Metric to optimize for
        metric_direction: "maximize" or "minimize"

    Returns:
        Tuple of (best_model_name, best_model_info)
    """
    # Filter out failed models
    valid_results = {
        name: info
        for name, info in results.items()
        if info.get("status") == "success" and info.get("metrics")
    }

    if not valid_results:
        raise ValueError("No successful model training results to compare")

    # Find best model
    if metric_direction == "maximize":
        best_name = max(
            valid_results.keys(),
            key=lambda m: valid_results[m]["metrics"][primary_metric],
        )
    else:
        best_name = min(
            valid_results.keys(),
            key=lambda m: valid_results[m]["metrics"][primary_metric],
        )

    logger.info(f"Best model: {best_name}")
    logger.info(
        f"Best {primary_metric}: {valid_results[best_name]['metrics'][primary_metric]:.4f}"
    )

    return best_name, valid_results[best_name]
