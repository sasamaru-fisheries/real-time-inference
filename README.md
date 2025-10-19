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
├── pmml-demo
│   ├── pom.xml
│   └── src
│       ├── main
│       │   └── java
│       │       ├── com/example/PMMLPredictor.java
│       │       └── com/example/OnnxPredictor.java
│       └── test (未使用)
└── pmml-standalone
    ├── libs
    ├── model
    │   └── model.pmml
    └── PMMLPredictor.java
```

- `README.md`  
  このドキュメント。プロジェクトの背景、実行手順、ディレクトリごとの説明を掲載しています。

- `train.py`  
  Python スクリプト。主な構成要素は次のとおりです。  
  - `build_pipeline` 関数: `RandomForestClassifier` を含む `PMMLPipeline` を生成。  
  - `export_pmml` 関数: Iris データを読み込み、学習 → `sklearn2pmml` で PMML を出力（`model/model.pmml` に保存）。  
  - `main` 関数: プロジェクトルートからの相対パスを計算し、学習と PMML 書き出しを実行。  
  実行すると `model/model.pmml` に最新モデルが書き出され、スタンドアロン用 (`pmml-standalone/model/model.pmml`) にもコピーされます。

- `pmml-demo/pom.xml`  
  Maven の設定ファイル。  
  - `<dependencies>` セクションで JPMML ライブラリや JAXB、Jackson を宣言。  
  - `<build>` 内で `maven-shade-plugin` を設定し、依存ライブラリ込みの fat JAR（`target/pmml-demo-1.0-SNAPSHOT.jar`）を作成。  
  - `exec-maven-plugin` により `mvn exec:java` で `com.example.PMMLPredictor` を直接起動できるように設定。

- `pmml-demo/src/main/java/com/example/PMMLPredictor.java`  
  Java のエントリーポイント。細部の役割は以下の通りです。  
  - `--model <path>` フラグ（または `--model=...`）で外部 PMML ファイルを指定。指定が無い場合は実行ディレクトリから `model/model.pmml` を探す。  
  - `loadEvaluator()` が指定ファイルを読み込み、`ModelEvaluatorBuilder` で推論器を初期化。  
  - `parseArguments()` でコマンドライン引数を 4 つの特徴量にマッピング。未指定なら Iris setosa の代表値を使用。  
  - `prepareArguments()` で JPMML の入力フィールドに対応する値へ変換。  
  - `printProbabilities()` で推論結果のクラス別確率を整形表示。  
  - `main()` では引数読み取り→推論→結果表示までを順に実行。

- `pmml-demo/src/main/java/com/example/OnnxPredictor.java`  
  ONNX Runtime を利用した推論クラス。  
  - `--model <path>` フラグで外部 ONNX ファイル（デフォルトは `model/model.onnx`）を指定。  
  - ONNX Runtime に入力テンソルを渡し、出力されたラベル・確率を整形表示。  
  - PMML と同じく順序付きの 4 特徴量（花の測定値）を CLI から受け付ける。

- `model/model.pmml`  
  `train.py` が生成する最新の PMML モデル本体（XML）。Java アプリはこのファイルを外部から読み込みます。

- `model/model.onnx`  
  `train.py` が生成する ONNX 形式のモデル。`OnnxPredictor` から読み込んで推論を行います。

- `pmml-demo/src/test`  
  Maven の標準構成で自動生成された空ディレクトリ。現在テストコードは置いていません。

- `pmml-demo/target/`  
  Maven がビルド時に作る作業ディレクトリ（`mvn package` 後に生成）。`pmml-demo-1.0-SNAPSHOT.jar` や中間ファイルが入ります。不要になったら削除しても問題ありません。

- `pmml-standalone/`  
  Maven を使わずに推論を実行するための手動セットアップ。  
  - `PMMLPredictor.java`: 依存 JAR を自分で指定してコンパイル・実行する形のスタンドアロン版。  
  - `model/model.pmml`: `train.py` からコピーされる最新モデル。  
  - `libs/`: `mvn dependency:copy-dependencies` で収集した外部ライブラリ群。Maven が使えない環境へ丸ごと持ち込めば、そのまま `javac` / `java` で実行可能。

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
   python3 -m pip install pandas scikit-learn sklearn2pmml
   ```

2. **PMML モデルの生成**

   ```bash
   python3 train.py
   ```

   成功すると `model/model.pmml` が更新され、最新モデルが保存されます（同時に `pmml-standalone/model/model.pmml` にもコピーされます）。
   さらに `model/model.onnx` も生成され、ONNX 推論で利用できます。

3. **Java アプリ (JAR) のビルド**

   ```bash
   cd pmml-demo
   mvn -q clean package
   ```

   `target/pmml-demo-1.0-SNAPSHOT.jar` が生成されます。依存ライブラリ込みの「fat JAR」なので、Java が動く環境ならどこでも単独で実行できます。

   > メモ: Maven から直接実行する場合は  
   > `mvn -q exec:java -Dexec.args="--model ../model/model.pmml"`  
   > のように `--model` オプションで外部 PMML のパスを指定してください。

4. **推論の実行 (デフォルト入力)**

   ```bash
   java -jar pmml-demo/target/pmml-demo-1.0-SNAPSHOT.jar --model model/model.pmml
   ```

   Iris setosa に近い標準入力で推論が行われ、クラス判定と確率が表示されます。

5. **推論の実行 (特徴量を指定)**

   ```bash
    java -jar pmml-demo/target/pmml-demo-1.0-SNAPSHOT.jar --model model/model.pmml 6.1 2.8 4.7 1.2
   ```

   引数は順に `sepal length`, `sepal width`, `petal length`, `petal width` です。  
   値を変えることで任意の測定値に対する予測結果が得られます。

6. **複数件をまとめて推論 (PMML)**

   ```bash
   java -jar pmml-demo/target/pmml-demo-1.0-SNAPSHOT.jar --model model/model.pmml --batch samples.txt
   ```

   `samples.txt` には 1 行につき 4 つの値（空白もしくはカンマ区切り）を記載します。`#` で始まる行や空行は無視されます。

---

### ONNX 版 CLI の実行

ONNX 推論は同じ JAR に含まれている `com.example.OnnxPredictor` を呼び出します。

```bash
# 既定値で推論
java -cp pmml-demo/target/pmml-demo-1.0-SNAPSHOT.jar com.example.OnnxPredictor --model model/model.onnx

# 特徴量を指定
java -cp pmml-demo/target/pmml-demo-1.0-SNAPSHOT.jar com.example.OnnxPredictor --model model/model.onnx 6.1 2.8 4.7 1.2

# 複数件を一括で推論
java -cp pmml-demo/target/pmml-demo-1.0-SNAPSHOT.jar com.example.OnnxPredictor --model model/model.onnx --batch samples.txt
```

`onnxruntime` のネイティブライブラリも fat JAR に含まれるため、追加セットアップは不要です。

---

## Maven を使わない実行方法

`pmml-standalone` ディレクトリには、必要な外部ライブラリ（JAR）を同梱した手動構成を用意しています。Maven が利用できない環境では次の手順で推論を実行できます。

```bash
cd pmml-standalone
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
pmml-standalone/
  ├── libs/         # 依存ライブラリ（PMML/ONNX 共通）
  ├── PMMLPredictor.java
  └── model/
       ├── model.pmml
       └── model.onnx (必要ならコピー)
```

1. **JDK の確認**  
   スタンドアロン構成は Java 17 を想定しています。`java -version` で確認してください。

2. **PMML 推論をビルド & 実行**

   ```bash
   cd pmml-standalone
   javac -cp "libs/*" PMMLPredictor.java
   java -cp ".:libs/*" PMMLPredictor --model model/model.pmml
   java -cp ".:libs/*" PMMLPredictor --model model/model.pmml 6.1 2.8 4.7 1.2
   ```

3. **ONNX 推論を行いたい場合**  
   `pmml-demo/src/main/java/com/example/OnnxPredictor.java` を同じフォルダにコピーしてコンパイルし実行します。

   ```bash
   cp ../pmml-demo/src/main/java/com/example/OnnxPredictor.java .
javac -cp "libs/*" OnnxPredictor.java
java -cp ".:libs/*" OnnxPredictor --model model/model.onnx
java -cp ".:libs/*" OnnxPredictor --model model/model.onnx 6.1 2.8 4.7 1.2
java -cp ".:libs/*" OnnxPredictor --model model/model.onnx --batch samples.txt
```

   `libs/` に `onnxruntime-*.jar` が含まれていれば、追加の `.so/.dylib/.dll` を用意する必要はありません。

4. **依存ライブラリを更新したい場合**  
   Maven が使える環境で一度 `mvn dependency:copy-dependencies -DincludeScope=compile -DoutputDirectory=pmml-standalone/libs` を実行すると `libs/` を再生成できます。

---

### 手動でライブラリを集める場合

Maven が使えない環境で依存 JAR を直接集める必要がある場合は、下表のリンクからダウンロードして `pmml-standalone/libs/` に配置してください（バージョンは必ず一致させる）。

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

---

## よくある質問

### Q. 新しいモデルを再学習したい場合は？
`python3 train.py` を再実行してください。`model/model.pmml`（および `pmml-standalone/model/model.pmml`）が更新されます。必要に応じて `mvn package` で新しいアプリケーション JAR を生成し、実行時は `--model` オプションで最新ファイルを指定します。

### Q. 別の環境（例: Databricks）で実行したい
生成された `pmml-demo-1.0-SNAPSHOT.jar` と `model` ディレクトリ（PMML/ONNX ファイル）をアップロードし、Java 実行環境で  
`java -jar pmml-demo-1.0-SNAPSHOT.jar --model /dbfs/.../model.pmml` または  
`java -cp pmml-demo-1.0-SNAPSHOT.jar com.example.OnnxPredictor --model /dbfs/.../model.onnx`  
のように実行してください。依存ライブラリは JAR に含まれているため追加インストールは不要です。

---

## ライセンス

このサンプルは自由に改変・利用できます。必要に応じてプロジェクトに合わせたライセンス表記をご検討ください。
