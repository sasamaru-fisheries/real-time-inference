"""Train an Iris classifier and export it as PMML and ONNX for Java predictors."""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def build_pipeline(random_state: int = 42) -> PMMLPipeline:
    """Create the PMML-ready scikit-learn pipeline."""
    return PMMLPipeline([
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=random_state))
    ])


def train_pipeline() -> tuple[PMMLPipeline, list[str]]:
    """Fit the PMML pipeline on the Iris dataset and return the fitted pipeline and feature names."""
    iris = load_iris()
    feature_names = list(iris.feature_names)
    features = pd.DataFrame(iris.data, columns=feature_names)
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

    return pipeline, feature_names


def export_pmml(pipeline: PMMLPipeline, output_path: Path) -> None:
    """Export the fitted pipeline to PMML format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sklearn2pmml(pipeline, str(output_path), with_repr=True)
    print(f"PMML model exported to: {output_path}")


def export_onnx(pipeline: PMMLPipeline, feature_names: list[str], output_path: Path) -> None:
    """Export the fitted classifier to ONNX format."""
    classifier = pipeline.named_steps["classifier"]
    initial_type = [("input", FloatTensorType([None, len(feature_names)]))]
    onnx_model = convert_sklearn(
        classifier,
        initial_types=initial_type,
        target_opset=17,
        options={"zipmap": False},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"ONNX model exported to: {output_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    models_dir = project_root / "model"
    pipeline, feature_names = train_pipeline()

    pmml_output = models_dir / "model.pmml"
    export_pmml(pipeline, pmml_output)

    onnx_output = models_dir / "model.onnx"
    export_onnx(pipeline, feature_names, onnx_output)

    # Update the standalone folder copy for Maven-less execution.
    standalone_path = project_root / "pmml-standalone" / "model" / "model.pmml"
    standalone_path.parent.mkdir(parents=True, exist_ok=True)
    standalone_path.write_bytes(pmml_output.read_bytes())
    print(f"PMML model copied to standalone directory: {standalone_path}")


if __name__ == "__main__":
    main()
