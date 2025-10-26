# PMML Demo (Iris Classifier)

このリポジトリは、Python で学習した機械学習モデルを PMML 形式で保存し、Java アプリ (=CLI) から推論を実行する最小構成のサンプルです。  
「Python でモデルを作る → Java で推論する」流れを理解したい Java 初心者の方を想定しています。

---

## プロジェクト構成

```
.
├── README.md
├── train.py
├── model
│   ├── model.pmml
│   └── model.onnx
├── pmml-predictor
│   ├── pom.xml
│   └── src
│       ├── main
│       │   └── java
│       │       └── com/example/PMMLPredictor.java
│       └── test (未使用)
├── onnx-predictor
│   ├── pom.xml
│   └── src
│       └── main
│           └── java
│               └── com/example/OnnxPredictor.java
├── standalone-pmml
│   ├── libs
│   ├── model
│   │   └── model.pmml
│   └── PMMLPredictor.java
└── standalone-onnx
    ├── libs
    │   └── onnxruntime-*.jar
    ├── model
    │   └── model.onnx
    └── OnnxPredictor.java
```

- `README.md`  
  このドキュメント。プロジェクトの背景、実行手順、ディレクトリごとの説明を掲載しています。

- `train.py`  
  Python スクリプト。主な構成要素は次のとおりです。  
  - `build_pipeline` 関数: `RandomForestClassifier` を含む `PMMLPipeline` を生成。  
    標準化（`StandardScaler`）を前段に挟んでおり、推論時にも同じ前処理が適用されます。  
  - `export_pmml` 関数: Iris データを読み込み、学習 → `sklearn2pmml` で PMML を出力（`model/model.pmml` に保存）。  
  - `main` 関数: プロジェクトルートからの相対パスを計算し、学習と PMML 書き出しを実行。  
  実行すると `model/model.pmml` / `model/model.onnx` に最新モデルが書き出され、スタンドアロン用 (`standalone-pmml/model/model.pmml`, `standalone-onnx/model/model.onnx`) にもコピーされます。

- `pmml-predictor/pom.xml`  
  Maven の設定ファイル。  
  - `<dependencies>` セクションで JPMML ライブラリや JAXB、Jackson を宣言。  
  - `<build>` 内で `maven-shade-plugin` を設定し、依存ライブラリ込みの fat JAR（`target/pmml-predictor-1.0-SNAPSHOT.jar`）を作成。  
  - `exec-maven-plugin` により `mvn exec:java` で `com.example.PMMLPredictor` を直接起動できるように設定。

- `pmml-predictor/src/main/java/com/example/PMMLPredictor.java`  
  Java のエントリーポイント。細部の役割は以下の通りです。  
  - `--model <path>` フラグ（または `--model=...`）で外部 PMML ファイルを指定。指定が無い場合は実行ディレクトリから `model/model.pmml` を探す。  
  - `loadEvaluator()` が指定ファイルを読み込み、`ModelEvaluatorBuilder` で推論器を初期化。  
  - `parseArguments()` でコマンドライン引数を 4 つの特徴量にマッピング。未指定なら Iris setosa の代表値を使用。  
  - `prepareArguments()` で JPMML の入力フィールドに対応する値へ変換。  
  - `printProbabilities()` で推論結果のクラス別確率を整形表示。  
  - `main()` では引数読み取り→推論→結果表示までを順に実行。

- `onnx-predictor/pom.xml`  
  ONNX 用の最小 Maven プロジェクト。`onnxruntime` だけを依存として取り込み、`com.example.OnnxPredictor` をエントリーポイントに設定します。

- `onnx-predictor/src/main/java/com/example/OnnxPredictor.java`  
  ONNX Runtime を利用した推論クラス。`--model` で ONNX ファイルを指定し、PMML 版と同じ引数／バッチ形式で推論を実行できます。

- `model/model.pmml`  
  `train.py` が生成する最新の PMML モデル本体（XML）。Java アプリはこのファイルを外部から読み込みます。

- `model/model.onnx`  
  `train.py` が生成する ONNX 形式のモデル。`OnnxPredictor` から読み込んで推論を行います。

- `pmml-predictor/src/test`  
  Maven の標準構成で自動生成された空ディレクトリ。現在テストコードは置いていません。

- `pmml-predictor/target/` および `onnx-predictor/target/`  
  Maven がビルド時に作る作業ディレクトリ（`mvn package` 後に生成）。`pmml-predictor-1.0-SNAPSHOT.jar` / `onnx-predictor-1.0-SNAPSHOT.jar` や中間ファイルが入ります。不要になったら削除しても問題ありません。

- `standalone-pmml/`  
  Maven を使わずに PMML 推論を実行するための構成。`PMMLPredictor.java` と JPMML 系 JAR（`libs/`）だけを含みます。

- `standalone-onnx/`  
  Maven を使わずに ONNX 推論を実行するための構成。`OnnxPredictor.java` と `onnxruntime` JAR（`libs/`）のみを含みます。

---

## 前提ソフトウェア

- Python 3.9 以上（pandas / scikit-learn / sklearn2pmml を利用します）
- Java 17
- Maven 3.9 以上

macOS や Linux であれば、Homebrew / apt などでインストールしておきます。

---

## セットアップ手順

1. **Python 依存関係のインストール**

   ```bash
   python3 -m pip install pandas scikit-learn sklearn2pmml skl2onnx packaging
   ```

2. **PMML モデルの生成**

   ```bash
   python3 train.py
   ```

   成功すると `model/model.pmml` が更新され、最新モデルが保存されます（同時に `standalone-pmml/model/model.pmml` にもコピーされます）。
   さらに `model/model.onnx` も生成され、ONNX 推論で利用できます。

3. **PMML 用 JAR のビルド**

   ```bash
   cd pmml-predictor
   mvn -q clean package
   cd ..
   ```

   `pmml-predictor/target/pmml-predictor-1.0-SNAPSHOT.jar` が生成されます。依存ライブラリ込みの「fat JAR」なので、Java が動く環境ならどこでも単独で実行できます。

4. **PMML 推論 (デフォルト入力)**

   ```bash
   java -jar pmml-predictor/target/pmml-predictor-1.0-SNAPSHOT.jar --model model/model.pmml
   ```

   Iris setosa に近い標準入力で推論が行われ、クラス判定と確率が表示されます。

5. **PMML 推論 (特徴量を指定)**

   ```bash
   java -jar pmml-predictor/target/pmml-predictor-1.0-SNAPSHOT.jar --model model/model.pmml 6.1 2.8 4.7 1.2
   ```

   引数は順に `sepal length`, `sepal width`, `petal length`, `petal width` です。  
   値を変えることで任意の測定値に対する予測結果が得られます。

6. **複数件をまとめて推論 (PMML)**

   ```bash
   java -jar pmml-predictor/target/pmml-predictor-1.0-SNAPSHOT.jar --model model/model.pmml --batch samples.txt
   ```

   `samples.txt` には 1 行につき 4 つの値（空白もしくはカンマ区切り）を記載します。`#` で始まる行や空行は無視されます。

---

### ONNX 版 CLI の実行

ONNX 推論は同じ JAR に含まれている `com.example.OnnxPredictor` を呼び出します。

事前に以下で JAR を作成してください:

```bash
cd onnx-predictor
mvn -q clean package
cd ..
```

```bash
# 既定値で推論
java -jar onnx-predictor/target/onnx-predictor-1.0-SNAPSHOT.jar --model model/model.onnx

# 特徴量を指定
java -jar onnx-predictor/target/onnx-predictor-1.0-SNAPSHOT.jar --model model/model.onnx 6.1 2.8 4.7 1.2

# 複数件を一括で推論
java -jar onnx-predictor/target/onnx-predictor-1.0-SNAPSHOT.jar --model model/model.onnx --batch samples.txt
```

`onnxruntime` のネイティブライブラリも fat JAR に含まれるため、追加セットアップは不要です。

---

## Maven を使わない実行方法

`standalone-pmml` / `standalone-onnx` には、それぞれの推論方式に必要な JAR を同梱した構成を用意しています。Maven が利用できない環境では次の手順で推論を実行できます（以下は PMML 版の例）。

```bash
cd standalone-pmml
# 必要なら model/model.pmml を最新に差し替える（train.py の出力をコピー）
javac -cp "libs/*" PMMLPredictor.java
java -cp ".:libs/*" PMMLPredictor              # 既定値で推論
java -cp ".:libs/*" PMMLPredictor --model model/model.pmml 6.1 2.8 4.7 1.2  # 引数付き
java -cp ".:libs/*" PMMLPredictor --model model/model.pmml --batch samples.txt
```

`libs/` 以下の JAR を実行環境にまとめて持ち込めば、JDK だけで同じ予測を再現できます。

---

## standalone 実行（Maven なし）の詳細手順

Maven が使えないマシンで推論を行いたい場合は、以下のファイル一式をコピーしてください。

```
standalone-pmml/
  ├── libs/         # JPMML 関連のみ
  ├── PMMLPredictor.java
  └── model/
       └── model.pmml
standalone-onnx/
  ├── libs/         # onnxruntime のみ
  ├── OnnxPredictor.java
  └── model/
       └── model.onnx
```

1. **JDK の確認**  
   スタンドアロン構成は Java 17 を想定しています。`java -version` で確認してください。

2. **PMML 推論をビルド & 実行**

   ```bash
   cd standalone-pmml
   javac -cp "libs/*" PMMLPredictor.java
   java -cp ".:libs/*" PMMLPredictor --model model/model.pmml
   java -cp ".:libs/*" PMMLPredictor --model model/model.pmml 6.1 2.8 4.7 1.2
   ```

3. **ONNX 推論を行いたい場合**  
   `standalone-onnx` ディレクトリへ移動し、同梱の `OnnxPredictor.java` と `onnxruntime` のみでコンパイル・実行します。

   ```bash
   cd standalone-onnx
   javac -cp "libs/*" OnnxPredictor.java
   java -cp ".:libs/*" OnnxPredictor --model model/model.onnx
   java -cp ".:libs/*" OnnxPredictor --model model/model.onnx 6.1 2.8 4.7 1.2
   java -cp ".:libs/*" OnnxPredictor --model model/model.onnx --batch samples.txt
   ```

   `libs/` に `onnxruntime-*.jar` が含まれていれば、追加の `.so/.dylib/.dll` を用意する必要はありません。

4. **依存ライブラリを更新したい場合**  
   Maven が使える環境で一度  
   `mvn dependency:copy-dependencies -DincludeScope=compile -DoutputDirectory=standalone-pmml/libs`  
   `mvn dependency:copy-dependencies -DincludeScope=compile -DoutputDirectory=standalone-onnx/libs -DincludeArtifactIds=onnxruntime`  
   を実行すると、それぞれの `libs/` を再生成できます。

---

### 手動でライブラリを集める場合

Maven が使えない環境で依存 JAR を直接集める必要がある場合は、下表のリンクからダウンロードして `standalone-pmml/libs/`（PMML 用）や `standalone-onnx/libs/`（ONNX 用）に配置してください（バージョンは必ず一致させる）。

| ライブラリ | 役割 | ダウンロードリンク |
| --- | --- | --- |
| `org.jpmml:pmml-evaluator:1.5.15` | PMML 推論エンジン | https://repo1.maven.org/maven2/org/jpmml/pmml-evaluator/1.5.15/pmml-evaluator-1.5.15.jar |
| `org.jpmml:pmml-model:1.5.15` | PMML モデル定義の読み書き | https://repo1.maven.org/maven2/org/jpmml/pmml-model/1.5.15/pmml-model-1.5.15.jar |
| `org.glassfish.jaxb:jaxb-runtime:2.3.3` | JAXB 実行時（XML バインディング） | https://repo1.maven.org/maven2/org/glassfish/jaxb/jaxb-runtime/2.3.3/jaxb-runtime-2.3.3.jar |
| `javax.xml.bind:jaxb-api:2.3.1` | JAXB API 定義 | https://repo1.maven.org/maven2/javax/xml/bind/jaxb-api/2.3.1/jaxb-api-2.3.1.jar |
| `jakarta.xml.bind:jakarta.xml.bind-api:2.3.3` | 最新 JAXB API（互換補助） | https://repo1.maven.org/maven2/jakarta/xml/bind/jakarta.xml.bind-api/2.3.3/jakarta.xml.bind-api-2.3.3.jar |
| `com.fasterxml.jackson.core:jackson-core:2.17.2` | JSON 基盤処理 | https://repo1.maven.org/maven2/com/fasterxml/jackson/core/jackson-core/2.17.2/jackson-core-2.17.2.jar |
| `com.fasterxml.jackson.core:jackson-databind:2.17.2` | JSON <-> オブジェクト変換 | https://repo1.maven.org/maven2/com/fasterxml/jackson/core/jackson-databind/2.17.2/jackson-databind-2.17.2.jar |
| `com.fasterxml.jackson.core:jackson-annotations:2.17.2` | Jackson アノテーション定義 | https://repo1.maven.org/maven2/com/fasterxml/jackson/core/jackson-annotations/2.17.2/jackson-annotations-2.17.2.jar |
| `com.microsoft.onnxruntime:onnxruntime:1.19.2` | ONNX 推論ランタイム（ネイティブ同梱） | https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime/1.19.2/onnxruntime-1.19.2.jar |
| `com.google.guava:guava:30.1-jre` | ユーティリティ（JPMML が利用） | https://repo1.maven.org/maven2/com/google/guava/guava/30.1-jre/guava-30.1-jre.jar |
| `com.google.guava:failureaccess:1.0.1` | Guava の補助パッケージ | https://repo1.maven.org/maven2/com/google/guava/failureaccess/1.0.1/failureaccess-1.0.1.jar |
| `org.apache.commons:commons-math3:3.6.1` | 数値計算ユーティリティ | https://repo1.maven.org/maven2/org/apache/commons/commons-math3/3.6.1/commons-math3-3.6.1.jar |
| `com.sun.istack:istack-commons-runtime:3.0.11` | JAXB 下位ユーティリティ | https://repo1.maven.org/maven2/com/sun/istack/istack-commons-runtime/3.0.11/istack-commons-runtime-3.0.11.jar |
| `javax.activation:javax.activation-api:1.2.0` | MIME/Activation API | https://repo1.maven.org/maven2/javax/activation/javax.activation-api/1.2.0/javax.activation-api-1.2.0.jar |
| `org.glassfish.jaxb:txw2:2.3.3` | JAXB の XML Writer コンポーネント | https://repo1.maven.org/maven2/org/glassfish/jaxb/txw2/2.3.3/txw2-2.3.3.jar |

ダウンロード後は `libs/` 配下にまとめ、`javac -cp "libs/*"` / `java -cp ".:libs/*"` でクラスパス指定すれば Maven なしでも推論可能です。

---

## Java 初心者向け補足

- **Maven とは？**  
  Java のビルド＆依存管理ツールです。`pom.xml` に書いた設定に従ってライブラリをダウンロードし、コンパイルから JAR 作成まで自動化します。

- **`mvn package` の挙動**  
  1. `src/main/java` をコンパイルして `target/classes` にクラスファイルを生成  
  2. `maven-shade-plugin` の設定により依存ライブラリをまとめた fat JAR を生成  
  3. マニフェストにエントリーポイント (`com.example.PMMLPredictor`) を書き込む  
     ※ モデル (`model/model.pmml`) は外部ファイルとして扱うため JAR には含めません。

- **PMML とは？**  
  Predictive Model Markup Language の略で、機械学習モデルを XML 形式で表現する標準仕様です。`sklearn2pmml` を使うと scikit-learn のモデルを簡単に PMML に変換でき、Java から `JPMML` ライブラリ経由で読み込めます。

- **コマンド例**  
  Maven や Java のコマンドは、すべてターミナル（macOS なら Terminal.app、Windows なら PowerShell等）で実行します。

- **推奨バージョンとダウンロード先**  
  - JDK 17 (LTS): Eclipse Temurin 公式配布 – https://adoptium.net/temurin/releases/?version=17  
  - Maven 3.9.x: Apache Maven Download – https://maven.apache.org/download.cgi  
    インストール手順の詳細は各公式ドキュメント（例: Maven の “Installing Apache Maven” ガイド）を参照してください。

---

## よくある質問

### Q. 新しいモデルを再学習したい場合は？
`python3 train.py` を再実行してください。`model/model.pmml`（および `standalone-pmml/model/model.pmml`）と `model/model.onnx`（および `standalone-onnx/model/model.onnx`）が更新されます。必要に応じて `mvn package` で新しいアプリケーション JAR を生成し、実行時は `--model` オプションで最新ファイルを指定します。

### Q. 別の環境（例: Databricks）で実行したい
生成された `pmml-predictor-1.0-SNAPSHOT.jar`（または `onnx-predictor-1.0-SNAPSHOT.jar`）と `model` ディレクトリをアップロードし、  
`java -jar pmml-predictor-1.0-SNAPSHOT.jar --model /dbfs/.../model.pmml` や  
`java -jar onnx-predictor-1.0-SNAPSHOT.jar --model /dbfs/.../model.onnx`  
のように実行してください。依存ライブラリは JAR に含まれているため追加インストールは不要です。

---

## ライセンス

このサンプルは自由に改変・利用できます。必要に応じてプロジェクトに合わせたライセンス表記をご検討ください。
