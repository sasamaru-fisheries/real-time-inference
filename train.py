"""Train an Iris classifier and export it as PMML for the Java predictor."""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline


def build_pipeline(random_state: int = 42) -> PMMLPipeline:
    """Create the PMML-ready scikit-learn pipeline."""
    return PMMLPipeline([
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=random_state))
    ])


def export_pmml(output_path: Path) -> None:
    """Train the model on the Iris dataset and export it to the given path."""
    iris = load_iris()
    features = pd.DataFrame(iris.data, columns=iris.feature_names)
    target = pd.Series(iris.target, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sklearn2pmml(pipeline, str(output_path), with_repr=True)
    print(f"PMML model exported to: {output_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    pmml_output = project_root / "pmml-demo" / "src" / "main" / "resources" / "model.pmml"
    export_pmml(pmml_output)


if __name__ == "__main__":
    main()
