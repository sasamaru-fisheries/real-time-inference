import argparse
from pathlib import Path

import joblib
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
TARGET = "Survived"


def build_pipeline(random_state: int, params: dict) -> Pipeline:
    numeric_features = ["Age", "SibSp", "Parch", "Fare"]
    categorical_features = ["Pclass", "Sex", "Embarked"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        bootstrap=params["bootstrap"],
        n_jobs=-1,
        random_state=random_state,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    missing_cols = set(FEATURES + [TARGET]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset {path} is missing required columns: {missing_cols}")

    df = df.dropna(subset=[TARGET])
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)
    return X, y


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, label: str) -> None:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print(f"\n=== Evaluation on {label} set ===")
    print(classification_report(y_test, preds))
    print(f"ROC AUC: {roc_auc_score(y_test, proba):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RandomForest on Titanic dataset.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/huge_1M_titanic.csv"),
        help="Path to the Titanic CSV file.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/random_forest_pipeline.pkl"),
        help="Where to store the trained model pipeline.",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        default=Path("data/huge_1M_titanic_2.csv"),
        help="External dataset used only for final evaluation.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for validation/test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=1,
        help="Number of Optuna trials for hyperparameter search.",
    )
    parser.add_argument(
        "--tune-sample-size",
        type=int,
        default=200_000,
        help="Number of rows to use during hyperparameter tuning (use entire dataset if smaller or if set to 0).",
    )
    return parser.parse_args()


def tune_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
    n_trials: int,
    sample_size: int,
) -> dict:
    if sample_size > 0 and len(X) > sample_size:
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=sample_size,
            stratify=y,
            random_state=random_state,
        )

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 4, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }

        pipeline = build_pipeline(random_state, params)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_valid)[:, 1]
        return roc_auc_score(y_valid, proba)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"Best trial ROC AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params


def main() -> None:
    args = parse_args()
    X_train_full, y_train_full = load_dataset(args.data)
    X_external, y_external = load_dataset(args.test_data)

    best_params = tune_hyperparameters(
        X_train_full,
        y_train_full,
        args.test_size,
        args.random_state,
        args.n_trials,
        args.tune_sample_size,
    )

    pipeline = build_pipeline(args.random_state, best_params)
    pipeline.fit(X_train_full, y_train_full)
    evaluate(pipeline, X_external, y_external, label="external test")

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_path)
    print(f"Saved RandomForest pipeline to {args.model_path}")


if __name__ == "__main__":
    main()
