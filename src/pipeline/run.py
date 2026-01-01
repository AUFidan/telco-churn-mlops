"""Pipeline orchestration for training and comparing all models."""

import argparse
import time
from pathlib import Path

import mlflow

from src.models.train import load_config, register_model, run_training
from src.pipeline.comparison import select_best_model
from src.pipeline.visualization import (
    create_metric_table,
    create_metrics_comparison_chart,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

AVAILABLE_MODELS = [
    "logistic_regression",
    "xgboost",
    "lightgbm",
    "catboost",
    "ensemble",
]


def train_all_models(models: list[str]) -> dict[str, dict]:
    """Train all specified models sequentially."""
    results = {}

    for model_name in models:
        logger.info("=" * 60)
        logger.info(f"Training model: {model_name}")
        logger.info("=" * 60)

        start_time = time.time()

        try:
            model, metrics, run_id = run_training(
                model_name=model_name,
                register=False,
            )

            training_time = time.time() - start_time

            results[model_name] = {
                "metrics": metrics,
                "run_id": run_id,
                "training_time": training_time,
                "status": "success",
            }

            logger.info(f"Completed {model_name} in {training_time:.1f}s")
            logger.info(f"Metrics: {metrics}")

        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            results[model_name] = {
                "metrics": None,
                "run_id": None,
                "training_time": time.time() - start_time,
                "status": "failed",
                "error": str(e),
            }

    return results


def run_pipeline(
    models: list[str] | None = None,
    primary_metric: str = "recall",
    register_best: bool = False,
    skip_ensemble: bool = False,
    output_dir: Path = Path("artifacts"),
) -> dict:
    """Run the full training pipeline for all models."""
    if models is None:
        models = AVAILABLE_MODELS.copy()

    if skip_ensemble and "ensemble" in models:
        models.remove("ensemble")

    logger.info(f"Starting pipeline with models: {models}")
    logger.info(f"Primary metric: {primary_metric}")

    # Setup MLflow
    config = load_config()
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Train all models
    results = train_all_models(models)

    # Extract metrics for visualization
    metrics_only = {
        name: info["metrics"]
        for name, info in results.items()
        if info["status"] == "success"
    }

    if not metrics_only:
        raise ValueError("All model training failed")

    # Create comparison table
    table = create_metric_table(metrics_only, primary_metric)
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    print(table)

    # Create comparison chart
    chart_path = create_metrics_comparison_chart(
        metrics_only,
        primary_metric=primary_metric,
        output_dir=output_dir / "plots",
    )

    # Select best model
    best_name, best_info = select_best_model(results, primary_metric)

    # Log comparison artifacts to MLflow
    with mlflow.start_run(run_name="pipeline_comparison"):
        mlflow.log_artifact(str(chart_path), artifact_path="comparison")
        mlflow.log_param("best_model", best_name)
        mlflow.log_param("primary_metric", primary_metric)
        mlflow.log_metrics({f"best_{k}": v for k, v in best_info["metrics"].items()})

    # Register best model if requested
    if register_best:
        registry_name = f"telco-churn-{best_name}"
        register_model(
            run_id=best_info["run_id"],
            model_name=registry_name,
            stage="Staging",
        )
        logger.info(f"Registered {best_name} to registry as {registry_name}")

    return {
        "results": results,
        "best_model": best_name,
        "best_metrics": best_info["metrics"],
        "chart_path": chart_path,
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run full training pipeline for Telco Churn models"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        choices=AVAILABLE_MODELS + ["all"],
        help="Models to train (default: all)",
    )
    parser.add_argument(
        "--primary-metric",
        type=str,
        default="recall",
        choices=["accuracy", "precision", "recall", "f1", "roc_auc"],
        help="Primary metric for model selection (default: recall)",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register best model to MLflow Model Registry",
    )
    parser.add_argument(
        "--skip-ensemble",
        action="store_true",
        help="Skip ensemble model (faster training)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Directory for output artifacts (default: artifacts)",
    )

    args = parser.parse_args()

    models = args.models
    if models is None or "all" in models:
        models = AVAILABLE_MODELS.copy()
        if args.skip_ensemble:
            models.remove("ensemble")

    result = run_pipeline(
        models=models,
        primary_metric=args.primary_metric,
        register_best=args.register,
        skip_ensemble=args.skip_ensemble,
        output_dir=Path(args.output_dir),
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Best model: {result['best_model']}")
    print(
        f"Best {args.primary_metric}: {result['best_metrics'][args.primary_metric]:.4f}"
    )
    print(f"Chart saved to: {result['chart_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
