# リアルタイム推論パイプライン（Titanic）

このレポジトリは Titanic 乗客データを題材に、Python で学習したモデルを Java から即座に推論できるようにするサンプルプロジェクトです。  
RandomForest と LightGBM の 2 つのパイプラインを用意しており、学習後は PMML / ONNX 形式にそのまま書き出します。

---

## レポジトリ構成

```
.
├── data/                         # 入力データ（Titanic-Dataset.csv など）
├── model/                        # エクスポートされた PMML / ONNX
├── models/                       # （参考）pickle 形式のモデルを保存していた旧ディレクトリ
├── reports/                      # 学習時に生成するレポート類
├── pmml-predictor/               # PMML モデルを読み込む Maven CLI
├── onnx-predictor/               # ONNX モデルを読み込む Maven CLI
├── standalone-pmml/              # 依存 JAR 同梱の PMML スタンドアロン CLI
├── standalone-onnx/              # 依存 JAR 同梱の ONNX スタンドアロン CLI
└── src/                          # Python 学習スクリプト
```

---

## Python での学習と評価

### 1. 依存関係のインストール

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` が無い場合は `pyproject.toml` の依存（pandas / scikit-learn / lightgbm / optuna / sklearn2pmml / skl2onnx など）を手動でインストールしてください。

### 2. RandomForest モデルの学習（PMML / ONNX 出力付き）

```bash
python src/train_random_forest.py \
  --data data/Titanic-Dataset.csv \
  --test-data data/Titanic-Dataset.csv \
  --tune-sample-size 0 \
  --report-dir reports/titanic/random_forest
```

学習後に自動で以下のファイルが生成されます。

- `model/titanic_random_forest.pmml`
- `model/titanic_random_forest.onnx`
- ROC 曲線や分類レポート（`reports/titanic/random_forest/` 配下）

### 3. LightGBM モデルの学習

```bash
python src/train_lightgbm.py \
  --data data/Titanic-Dataset.csv \
  --test-data data/Titanic-Dataset.csv \
  --tune-sample-size 0 \
  --report-dir reports/titanic/lightgbm
```

こちらも学習後に自動で以下を出力します。

- `model/titanic_lightgbm.pmml`
- `model/titanic_lightgbm.onnx`
- ROC 曲線や分類レポート（`reports/titanic/lightgbm/` 配下）

※ モデル書き出し先は `--pmml-path` / `--onnx-path` オプションで変更可能です。  
※ 学習かつエクスポートだけを行いたい場合は `--skip-pmml` / `--skip-onnx` を指定できます。

---

## Java CLI での推論

### PMML（Maven プロジェクト）

1. 依存 JAR（JPMML など）は `mvn clean package` で自動取得できます。

   ```bash
   cd pmml-predictor
   mvn -q clean package
   cd ..
   ```

2. 推論を実行

   ```bash
   java -jar pmml-predictor/target/pmml-predictor-1.0-SNAPSHOT.jar \
     --model model/titanic_random_forest.pmml \
     --batch data/sample_batch.txt
   ```

   - `--watch` を付与すると PMML ファイルを監視し、更新を検知して自動リロードします。
   - 引数を指定すれば 1 件のみを CLI から直接入力することも可能です。

### ONNX（Maven プロジェクト）

1. OS 依存をなくすため、`onnxruntime-platform-1.19.2.jar` を手元の `libs/` に配置します。

   ```bash
   (cd standalone-onnx && mvn -q dependency:copy-dependencies \
     -DincludeGroupIds=com.microsoft.onnxruntime \
     -DincludeArtifactIds=onnxruntime \
     -DoutputDirectory=libs \
     && mv -f libs/onnxruntime-1.19.2.jar libs/onnxruntime-platform-1.19.2.jar)

   cp standalone-onnx/libs/onnxruntime-platform-1.19.2.jar onnx-predictor/libs/
   ```

2. `onnx-predictor` をビルド

   ```bash
   cd onnx-predictor
   mvn -q clean package
   cd ..
   ```

3. 推論を実行

   ```bash
   java -jar onnx-predictor/target/onnx-predictor-1.0-SNAPSHOT.jar \
     --model model/titanic_random_forest.onnx \
     --batch data/sample_batch.txt
   ```

   既定では Iris 用の特徴量順になっているため、Titanic モデルを使う場合は入力整形を合わせるか `standalone-onnx/OnnxPredictor.java` を調整してください。

### スタンドアロン版（PMML / ONNX）

`standalone-pmml`・`standalone-onnx` ディレクトリは Maven を使わずに実行可能です。  
依存ライブラリをダウンロードした上で、`java -cp ".:libs/*" ...` と実行してください。

```bash
(cd standalone-pmml && mvn -q dependency:copy-dependencies \
  -DincludeGroupIds=org.jpmml,javax.xml.bind,com.fasterxml.jackson.core,org.apache.commons,com.google.guava,com.sun.istack \
  -DincludeArtifactIds=pmml-evaluator,pmml-model,jaxb-api,jaxb-runtime,jackson-core,jackson-databind,jackson-annotations,commons-math3,guava,failureaccess,istack-commons-runtime,javax.activation-api,txw2 \
  -DoutputDirectory=libs)

(cd standalone-onnx && mvn -q dependency:copy-dependencies \
  -DincludeGroupIds=com.microsoft.onnxruntime \
  -DincludeArtifactIds=onnxruntime \
  -DoutputDirectory=libs \
  && mv -f libs/onnxruntime-1.19.2.jar libs/onnxruntime-platform-1.19.2.jar)
```

---

## 運用パターン別の比較

| パターン | モデル形式 | 追加バイナリ | 特徴 | 備考 |
| --- | --- | --- | --- | --- |
| Maven + PMML (`pmml-predictor`) | `titanic_random_forest.pmml` | JAR 約 10MB | 純 Java 実装で軽量 | `--watch` でホットリロード可 |
| Maven + ONNX (`onnx-predictor`) | `titanic_random_forest.onnx` 等 | JAR 約 90MB（`onnxruntime-platform` 同梱） | ネイティブ最適化で高速 | `libs/onnxruntime-platform-1.19.2.jar` を手動配置 |
| Standalone + PMML | 同上 | `libs/` 約 10MB | Java ランタイムのみで実行可能 | スクリプトから javac/ java 実行 |
| Standalone + ONNX | `*.onnx` | `libs/` 約 89MB（`onnxruntime-platform`） | Maven 版と同等 | 配布時に OS 依存が少ない jar を同梱 |

---

## 補足

- `data/` 配下には学習・推論に使うサンプルデータを配置しています。行形式の `sample_batch.txt` をそのまま CLI に渡せます。
- `reports/` フォルダは git ignore に含まれているため、生成されたレポートはコミットされません。
- 旧来の PMML エクスポートスクリプト（`src/export_to_pmml.py`）は撤廃し、学習スクリプトが直接 PMML / ONNX を出力します。
- ONNX 変換では LightGBM コンバーターを追加登録しているため、`onnxmltools` が必須です。
- OS ごとのランタイム配布が必要な場合は、`onnxruntime-platform` の jar を別途配布するだけで済みます（ネイティブライブラリ同梱済み）。

---

## ライセンス

このリポジトリのコードは MIT ライセンスを想定しています（必要に応じて `LICENSE` ファイルを参照してください）。
