"""Utility helpers to export trained scikit-learn pipelines to PMML and ONNX."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype
from sklearn.pipeline import Pipeline

from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common._registration import get_shape_calculator
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType, StringTensorType
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,
    calculate_linear_regressor_output_shapes,
)


def export_pipeline_to_pmml(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    features: Sequence[str],
    numeric_features: Iterable[str],
    target: str,
    output_path: Path,
) -> None:
    """Export a fitted pipeline to PMML, re-fitting on the provided dataset."""
    numeric_features = set(numeric_features)
    X_pmml = X[list(features)].copy()
    y_pmml = y.loc[X_pmml.index]

    for col in features:
        if col in numeric_features:
            X_pmml[col] = pd.to_numeric(X_pmml[col], errors="coerce")
        else:
            X_pmml[col] = X_pmml[col].astype(str)

    # Drop rows with unresolved values to avoid PMML validation errors.
    valid_idx = X_pmml.dropna().index
    X_pmml = X_pmml.loc[valid_idx]
    y_pmml = y_pmml.loc[valid_idx]

    pmml_pipeline = PMMLPipeline([("pipeline", pipeline)])
    pmml_pipeline.active_fields = list(features)
    pmml_pipeline.target_fields = [target]
    pmml_pipeline.fit(X_pmml, y_pmml)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sklearn2pmml(pmml_pipeline, output_path, with_repr=True)
    print(f"Exported PMML model to {output_path}")


def export_pipeline_to_onnx(
    pipeline: Pipeline,
    X: pd.DataFrame,
    *,
    output_path: Path,
    target_opset: int | None = None,
    float_feature_names: Iterable[str] | None = None,
    string_feature_names: Iterable[str] | None = None,
) -> None:
    """Export a fitted pipeline to ONNX using the data frame schema."""
    _ensure_lightgbm_registration()
    float_feature_names = set(float_feature_names or [])
    string_feature_names = set(string_feature_names or [])
    initial_types: list[tuple[str, object]] = []
    for column in X.columns:
        if column in string_feature_names:
            tensor = StringTensorType([None, 1])
        elif column in float_feature_names or is_float_dtype(X[column]):
            tensor = FloatTensorType([None, 1])
        elif is_integer_dtype(X[column]):
            tensor = Int64TensorType([None, 1])
        else:
            tensor = StringTensorType([None, 1])
        initial_types.append((column, tensor))

    options = _build_onnx_options(pipeline)
    opset = _normalise_opset(target_opset, pipeline)
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_types,
        target_opset=opset,
        options=options,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Exported ONNX model to {output_path}")


def _ensure_lightgbm_registration() -> None:
    if getattr(_ensure_lightgbm_registration, "_done", False):
        return
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
    except ImportError:
        return

    try:
        get_shape_calculator(LGBMClassifier)
    except ValueError:
        update_registered_converter(
            LGBMClassifier,
            "LightGbmLGBMClassifier",
            calculate_linear_classifier_output_shapes,
            convert_lightgbm,
            options={"zipmap": [True, False], "nocl": [True, False]},
        )

    try:
        get_shape_calculator(LGBMRegressor)
    except ValueError:
        update_registered_converter(
            LGBMRegressor,
            "LightGbmLGBMRegressor",
            calculate_linear_regressor_output_shapes,
            convert_lightgbm,
            options={"zipmap": [True, False]},
        )

    _ensure_lightgbm_registration._done = True


def _build_onnx_options(pipeline: Pipeline) -> dict | None:
    last_step = None
    if hasattr(pipeline, "steps") and pipeline.steps:
        last_step = pipeline.steps[-1][1]

    if last_step is None:
        return None

    class_name = last_step.__class__.__name__
    if class_name.startswith("LGBM"):
        return {id(last_step): {"zipmap": False, "nocl": False}}
    return None


def _normalise_opset(target_opset: int | dict | None, pipeline: Pipeline) -> dict:
    if target_opset is None:
        opset: dict[str, int] = {"": 17}
    elif isinstance(target_opset, int):
        opset = {"": target_opset}
    else:
        opset = dict(target_opset)

    requires_ml = False
    if hasattr(pipeline, "steps") and pipeline.steps:
        last_step = pipeline.steps[-1][1]
        if last_step.__class__.__name__.startswith("LGBM"):
            requires_ml = True

    if requires_ml and "ai.onnx.ml" not in opset:
        opset["ai.onnx.ml"] = 3

    return opset
