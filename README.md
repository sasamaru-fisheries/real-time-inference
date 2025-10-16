# PMML Demo (Iris Classifier)

このリポジトリは、Python で学習した機械学習モデルを PMML 形式で保存し、Java アプリ (=CLI) から推論を実行する最小構成のサンプルです。  
「Python でモデルを作る → Java で推論する」流れを理解したい Java 初心者の方を想定しています。

---

## プロジェクト構成

```
.
├── README.md
├── train.py
└── pmml-demo
    ├── pom.xml
    └── src
        ├── main
        │   ├── java
        │   │   └── com/example/PMMLPredictor.java
        │   └── resources
        │       └── model.pmml
        └── test (未使用)
```

- `README.md`  
  このドキュメント。プロジェクトの背景、実行手順、ディレクトリごとの説明を掲載しています。

- `train.py`  
  Python スクリプト。主な構成要素は次のとおりです。  
  - `build_pipeline` 関数: `RandomForestClassifier` を含む `PMMLPipeline` を生成。  
  - `export_pmml` 関数: Iris データを読み込み、学習 → `sklearn2pmml` で PMML を出力。出力先は `pmml-demo/src/main/resources/model.pmml`。  
  - `main` 関数: プロジェクトルートからの相対パスを計算し、学習と PMML 書き出しを実行。  
  実行すると `PMML model exported to: ...` が表示され、Java 側のリソースに上書き保存されます。

- `pmml-demo/pom.xml`  
  Maven の設定ファイル。  
  - `<dependencies>` セクションで JPMML ライブラリや JAXB、Jackson を宣言。  
  - `<build>` 内で `maven-shade-plugin` を設定し、依存ライブラリ込みの fat JAR（`target/pmml-demo-1.0-SNAPSHOT.jar`）を作成。  
  - `exec-maven-plugin` により `mvn exec:java` で `com.example.PMMLPredictor` を直接起動できるように設定。

- `pmml-demo/src/main/java/com/example/PMMLPredictor.java`  
  Java のエントリーポイント。細部の役割は以下の通りです。  
  - `loadEvaluator()` でクラスパス上の `model.pmml` を読み込み、`ModelEvaluatorBuilder` で推論器を初期化。  
  - `parseArguments()` でコマンドライン引数を 4 つの特徴量にマッピング。未指定なら Iris setosa の代表値を使用。  
  - `prepareArguments()` で JPMML の入力フィールドに対応する値へ変換。  
  - `printProbabilities()` で推論結果のクラス別確率を整形表示。  
  - `main()` では引数読み取り→推論→結果表示までを順に実行。

- `pmml-demo/src/main/resources/model.pmml`  
  `train.py` が生成する PMML モデル本体（XML）。Iris データセットのランダムフォレストが記述されています。`mvn package` 時に JAR にバンドルされ、`PMMLPredictor` から `/model.pmml` として参照されます。

- `pmml-demo/src/test`  
  Maven の標準構成で自動生成された空ディレクトリ。現在テストコードは置いていません。

- `pmml-demo/target/`  
  Maven がビルド時に作る作業ディレクトリ（`mvn package` 後に生成）。`pmml-demo-1.0-SNAPSHOT.jar` や中間ファイルが入ります。不要になったら削除しても問題ありません。

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

   成功すると `pmml-demo/src/main/resources/model.pmml` が更新され、最新モデルが保存されます。

3. **Java アプリ (JAR) のビルド**

   ```bash
   cd pmml-demo
   mvn -q clean package
   ```

   `target/pmml-demo-1.0-SNAPSHOT.jar` が生成されます。依存ライブラリ込みの「fat JAR」なので、Java が動く環境ならどこでも単独で実行できます。

4. **推論の実行 (デフォルト入力)**

   ```bash
   java -jar target/pmml-demo-1.0-SNAPSHOT.jar
   ```

   Iris setosa に近い標準入力で推論が行われ、クラス判定と確率が表示されます。

5. **推論の実行 (特徴量を指定)**

   ```bash
   java -jar target/pmml-demo-1.0-SNAPSHOT.jar 6.1 2.8 4.7 1.2
   ```

   引数は順に `sepal length`, `sepal width`, `petal length`, `petal width` です。  
   値を変えることで任意の測定値に対する予測結果が得られます。

---

## Java 初心者向け補足

- **Maven とは？**  
  Java のビルド＆依存管理ツールです。`pom.xml` に書いた設定に従ってライブラリをダウンロードし、コンパイルから JAR 作成まで自動化します。

- **`mvn package` の挙動**  
  1. `src/main/java` をコンパイルして `target/classes` にクラスファイルを生成  
  2. `src/main/resources` のファイル（ここでは `model.pmml`）をコピー  
  3. `maven-shade-plugin` の設定により、これらをすべて一つの `jar` にまとめる  
  4. マニフェストにエントリーポイント (`com.example.PMMLPredictor`) を書き込む

- **PMML とは？**  
  Predictive Model Markup Language の略で、機械学習モデルを XML 形式で表現する標準仕様です。`sklearn2pmml` を使うと scikit-learn のモデルを簡単に PMML に変換でき、Java から `JPMML` ライブラリ経由で読み込めます。

- **コマンド例**  
  Maven や Java のコマンドは、すべてターミナル（macOS なら Terminal.app、Windows なら PowerShell等）で実行します。

---

## よくある質問

### Q. 新しいモデルを再学習したい場合は？
`python3 train.py` を再実行してください。その後、`mvn package` を実行すると新しい PMML が JAR に同梱されます。

### Q. 別の環境（例: Databricks）で実行したい
生成された `pmml-demo-1.0-SNAPSHOT.jar` をアップロードし、Java 実行環境で `java -jar ...` を実行するだけで利用できます。依存ライブラリは JAR に含まれているため追加インストールは不要です。

---

## ライセンス

このサンプルは自由に改変・利用できます。必要に応じてプロジェクトに合わせたライセンス表記をご検討ください。
