"""Convert the trained RandomForest pipeline into a PMML file.

RandomForest モデルは pickle (`models/titanic/random_forest_pipeline.pkl`) として保存済みです。
このスクリプトでは、同じデータセットを使って PMML 用のパイプラインに再フィットし、
`model/titanic_random_forest.pmml` としてエクスポートします。
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
NUMERIC_FEATURES = ["Age", "SibSp", "Parch", "Fare"]
TARGET = "Survived"


def main() -> None:
    """エクスポート処理のエントリーポイント。"""
    data_path = ROOT_DIR / "data" / "Titanic-Dataset.csv"
    model_path = ROOT_DIR / "models" / "titanic" / "random_forest_pipeline.pkl"
    output_path = ROOT_DIR / "model" / "titanic_random_forest.pmml"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find trained model at {model_path}. Run titanic/train_random_forest.py first."
        )

    df = pd.read_csv(data_path)
    X = df[FEATURES].copy()
    y = df[TARGET]

    # PMML は各列のデータ型に厳密なため、ここで型を統一しておく。
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in set(FEATURES) - set(NUMERIC_FEATURES):
        X[col] = X[col].astype(str)

    # 欠損値が残っている行を落とす。列ごとの imputter では処理しきれないケースを防ぐため。
    X = X.dropna()
    y = y.loc[X.index]

    pipeline = joblib.load(model_path)

    # 学習済み pipeline を PMMLPipeline でラップ。
    # active_fields / target_fields を明示指定すると、エクスポート後のモデルで
    # 期待どおりの列順・ラベルが使われる。
    pmml_pipeline = PMMLPipeline([
        ("pipeline", pipeline),
    ])
    pmml_pipeline.active_fields = FEATURES
    pmml_pipeline.target_fields = [TARGET]

    # sklearn2pmml は pipeline に "学習済み" フラグが立っている必要がある。
    # 同じデータで fit し直すことで、前処理（OneHotEncoder 等）の内部状態を PMML に含められる。
    pmml_pipeline.fit(X, y)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sklearn2pmml(pmml_pipeline, output_path, with_repr=True)
    print(f"Exported PMML model to {output_path}")


if __name__ == "__main__":
    main()
