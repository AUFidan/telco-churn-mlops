"""Training script with MLflow tracking and Optuna hyperparameter tuning."""

from pathlib import Path

import mlflow
import optuna
import yaml
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score

from src.data.preprocessing import preprocess_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


def create_optuna_objective(
    model_name: str,
    model_config: dict,
    X_train,
    y_train,
    cv_folds: int,
    scoring: str,
):
    """Create Optuna objective function for a given model."""

    def objective(trial: optuna.Trial) -> float:
        # Build params from search space
        params = {}

        for param_name, param_config in model_config.get("search_space", {}).items():
            if param_config["type"] == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config["choices"]
                )
            elif param_config["type"] == "int":
                params[param_name] = trial.suggest_int(
                    param_name, param_config["low"], param_config["high"]
                )
            elif param_config["type"] == "float":
                log_scale = param_config.get("log", False)
                params[param_name] = trial.suggest_float(
                    param_name, param_config["low"], param_config["high"], log=log_scale
                )

        # Add fixed params
        params.update(model_config.get("fixed", {}))

        # Create model
        if model_name == "logistic_regression":
            model = LogisticRegression(**params)
        elif model_name == "xgboost":
            model = XGBClassifier(**params)
        elif model_name == "lightgbm":
            model = LGBMClassifier(**params)
        elif model_name == "catboost":
            model = CatBoostClassifier(**params)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Cross-validation score
        scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring)
        return scores.mean()

    return objective


def train_model(
    model_name: str,
    config: dict,
    X_train,
    y_train,
    X_test,
    y_test,
) -> tuple:
    """Train a model with Optuna hyperparameter tuning and MLflow logging."""
    logger.info(f"Training {model_name} with Optuna tuning")

    model_config = config["models"][model_name]
    optuna_config = config["optuna"]

    # Create Optuna study
    study = optuna.create_study(direction=optuna_config["direction"])

    # Create objective
    objective = create_optuna_objective(
        model_name=model_name,
        model_config=model_config,
        X_train=X_train,
        y_train=y_train,
        cv_folds=optuna_config["cv_folds"],
        scoring=optuna_config["scoring"],
    )

    # Optimize
    study.optimize(objective, n_trials=optuna_config["n_trials"], show_progress_bar=True)

    logger.info(f"Best trial: {study.best_trial.value:.4f}")
    logger.info(f"Best params: {study.best_trial.params}")

    # Train final model with best params
    best_params = {**study.best_trial.params, **model_config.get("fixed", {})}

    if model_name == "logistic_regression":
        final_model = LogisticRegression(**best_params)
    elif model_name == "xgboost":
        final_model = XGBClassifier(**best_params)
    elif model_name == "lightgbm":
        final_model = LGBMClassifier(**best_params)
    elif model_name == "catboost":
        final_model = CatBoostClassifier(**best_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    final_model.fit(X_train, y_train)

    # Evaluate on test set
    metrics = evaluate_model(final_model, X_test, y_test)
    logger.info(f"Test metrics: {metrics}")

    return final_model, best_params, metrics, study


def create_model(model_name: str, params: dict):
    """Create a model instance with given parameters."""
    if model_name == "logistic_regression":
        return LogisticRegression(**params)
    elif model_name == "xgboost":
        return XGBClassifier(**params)
    elif model_name == "lightgbm":
        return LGBMClassifier(**params)
    elif model_name == "catboost":
        return CatBoostClassifier(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_ensemble(
    config: dict,
    X_train,
    y_train,
    X_test,
    y_test,
) -> tuple:
    """Train ensemble model using StackingClassifier with tuned base models."""
    logger.info("Training ensemble with StackingClassifier")

    ensemble_config = config["models"]["ensemble"]
    optuna_config = config["optuna"]

    # Base model names (exclude logistic_regression as it's the meta-learner)
    base_model_names = ["xgboost", "lightgbm", "catboost"]

    # Train each base model with Optuna to get best params
    base_estimators = []
    all_best_params = {}

    for model_name in base_model_names:
        logger.info(f"Tuning {model_name} for ensemble...")
        model_config = config["models"][model_name]

        study = optuna.create_study(direction=optuna_config["direction"])
        objective = create_optuna_objective(
            model_name=model_name,
            model_config=model_config,
            X_train=X_train,
            y_train=y_train,
            cv_folds=optuna_config["cv_folds"],
            scoring=optuna_config["scoring"],
        )
        study.optimize(
            objective,
            n_trials=optuna_config["n_trials"] // 3,  # Fewer trials per model
            show_progress_bar=True,
        )

        best_params = {**study.best_trial.params, **model_config.get("fixed", {})}
        all_best_params[model_name] = best_params

        base_model = create_model(model_name, best_params)
        base_estimators.append((model_name, base_model))

        logger.info(f"{model_name} best CV score: {study.best_trial.value:.4f}")

    # Create meta-learner (final estimator)
    final_estimator_config = ensemble_config["final_estimator"]
    final_estimator = LogisticRegression(
        **final_estimator_config.get("params", {"random_state": 4141})
    )

    # Create and train StackingClassifier
    stacking_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=ensemble_config["cv"],
        stack_method="predict_proba",
        n_jobs=-1,
    )

    logger.info("Fitting StackingClassifier...")
    stacking_model.fit(X_train, y_train)

    # Evaluate
    metrics = evaluate_model(stacking_model, X_test, y_test)
    logger.info(f"Ensemble test metrics: {metrics}")

    return stacking_model, all_best_params, metrics


def run_training(model_name: str = "logistic_regression"):
    """Run full training pipeline for a model."""
    config = load_config()

    # Setup MLflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Load and preprocess data
    logger.info("Loading and preprocessing data")
    data = preprocess_pipeline(
        raw_data_path=config["data"]["raw_path"],
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    # Train with MLflow tracking
    with mlflow.start_run(run_name=f"{model_name}_optuna"):
        # Log config
        mlflow.log_params(
            {
                "model_name": model_name,
                "n_trials": config["optuna"]["n_trials"],
                "cv_folds": config["optuna"]["cv_folds"],
                "test_size": config["data"]["test_size"],
                "random_state": config["data"]["random_state"],
            }
        )

        if model_name == "ensemble":
            # Train ensemble
            model, all_best_params, metrics = train_ensemble(
                config=config,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
            # Log best params for each base model
            for base_name, params in all_best_params.items():
                for k, v in params.items():
                    mlflow.log_param(f"{base_name}_{k}", v)
        else:
            # Train single model
            model, best_params, metrics, study = train_model(
                model_name=model_name,
                config=config,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
            # Log best hyperparameters
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info(f"Run completed. Run ID: {mlflow.active_run().info.run_id}")

    return model, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="logistic_regression",
        choices=["logistic_regression", "xgboost", "lightgbm", "catboost", "ensemble"],
        help="Model to train",
    )
    args = parser.parse_args()

    run_training(model_name=args.model)
