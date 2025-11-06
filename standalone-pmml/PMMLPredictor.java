import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.Computable;
import org.jpmml.evaluator.Evaluator;
import org.jpmml.evaluator.FieldValue;
import org.jpmml.evaluator.InputField;
import org.jpmml.evaluator.ModelEvaluatorBuilder;
import org.jpmml.evaluator.OutputField;
import org.jpmml.evaluator.TargetField;
import org.jpmml.model.PMMLUtil;

/**
 * Minimal standalone PMML predictor for the Titanic RandomForest model.
 *
 * Compilation example:
 *   javac -cp "libs/*" PMMLPredictor.java
 * Execution example:
 *   java -cp ".:libs/*" PMMLPredictor --model ../model/titanic_random_forest.pmml --batch ../data/sample_batch.txt
 */
public final class PMMLPredictor {

    private static final String MODEL_OPTION = "--model";
    private static final String BATCH_OPTION = "--batch";
    private static final Path DEFAULT_MODEL_PATH = Path.of("../model", "titanic_random_forest.pmml");

    private static final String[] FEATURE_ORDER = {
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked"
    };

    private static final Set<String> NUMERIC_FEATURES = new LinkedHashSet<>(List.of(
        "Age",
        "SibSp",
        "Parch",
        "Fare"
    ));

    private static final Map<String, String> CLASS_LABELS = Map.of(
        "0", "not_survived",
        "1", "survived"
    );

    private static final Map<String, Object> DEFAULT_FEATURES = Map.of(
        "Pclass", "3",
        "Sex", "male",
        "Age", 22.0,
        "SibSp", 1.0,
        "Parch", 0.0,
        "Fare", 7.25,
        "Embarked", "S"
    );

    private PMMLPredictor() {
    }

    public static void main(String[] args) throws Exception {
        // 1. CLI 引数でモデルファイルと入力を解釈
        ParsedInput parsedInput = parseInput(args);
        Path resolvedModelPath = resolveModelPath(parsedInput.modelPath(), parsedInput.modelExplicit());
        // 2. PMML の Evaluator を取得
        Evaluator evaluator = loadEvaluator(resolvedModelPath);

        List<Map<String, Object>> featureVectors = parsedInput.featureVectors();
        System.out.println("Model loaded from: " + resolvedModelPath.toAbsolutePath());
        System.out.println();

        for (int i = 0; i < featureVectors.size(); i++) {
            Map<String, Object> features = featureVectors.get(i);
            if (featureVectors.size() > 1) {
                System.out.printf("=== Sample %d ===%n", i + 1);
            }
            try {
                Map<FieldName, FieldValue> arguments = prepareArguments(evaluator, features);
                Map<FieldName, ?> results = evaluator.evaluate(arguments);
                printInput(features);
                printResults(results, evaluator);
            } catch (Exception ex) {
                System.err.println("Failed to evaluate sample " + (i + 1) + ": " + ex.getMessage());
                ex.printStackTrace(System.err);
            }
            if (i < featureVectors.size() - 1) {
                System.out.println();
            }
        }
    }

    private static ParsedInput parseInput(String[] args) {
        Path modelPath = DEFAULT_MODEL_PATH;
        boolean explicitModelPath = false;
        Path batchPath = null;
        List<String> featureArgs = new ArrayList<>();

        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            if (MODEL_OPTION.equals(arg)) {
                if ((i + 1) >= args.length) {
                    throw new IllegalArgumentException("--model requires a file path argument.");
                }
                modelPath = Path.of(args[++i]);
                explicitModelPath = true;
            } else if (BATCH_OPTION.equals(arg)) {
                if ((i + 1) >= args.length) {
                    throw new IllegalArgumentException("--batch requires a file path argument.");
                }
                batchPath = Path.of(args[++i]);
            } else if (MODEL_OPTION.length() < arg.length() && arg.startsWith(MODEL_OPTION + "=")) {
                modelPath = Path.of(arg.substring((MODEL_OPTION + "=").length()));
                explicitModelPath = true;
            } else if (BATCH_OPTION.length() < arg.length() && arg.startsWith(BATCH_OPTION + "=")) {
                batchPath = Path.of(arg.substring((BATCH_OPTION + "=").length()));
            } else {
                featureArgs.add(arg);
            }
        }

        List<Map<String, Object>> featureVectors = new ArrayList<>();
        if (!featureArgs.isEmpty()) {
            featureVectors.add(parseFeatureVector(featureArgs.toArray(String[]::new)));
        }
        if (batchPath != null) {
            featureVectors.addAll(readBatchFile(batchPath));
        }
        if (featureVectors.isEmpty()) {
            featureVectors.add(new LinkedHashMap<>(DEFAULT_FEATURES));
        }
        return new ParsedInput(modelPath, featureVectors, explicitModelPath);
    }

    private static Path resolveModelPath(Path modelPath, boolean explicitModelPath) throws IOException {
        if (Files.exists(modelPath)) {
            return modelPath;
        }
        if (explicitModelPath || modelPath.isAbsolute()) {
            throw new IOException("PMML model file not found: " + modelPath.toAbsolutePath());
        }
        Path parentCandidate = Path.of("..").resolve(modelPath);
        if (Files.exists(parentCandidate)) {
            return parentCandidate;
        }
        throw new IOException(
            "PMML model file not found. Checked: " + modelPath.toAbsolutePath() + " and " + parentCandidate.toAbsolutePath());
    }

    private static Evaluator loadEvaluator(Path resolvedModelPath) throws Exception {
        PMML pmml;
        try (InputStream is = Files.newInputStream(resolvedModelPath)) {
            pmml = PMMLUtil.unmarshal(is);
        }
        Evaluator evaluator = new ModelEvaluatorBuilder(pmml).build();
        evaluator.verify();
        return evaluator;
    }

    private static Map<String, Object> parseFeatureVector(String[] tokens) {
        if (tokens.length != FEATURE_ORDER.length) {
            throw new IllegalArgumentException("Expected " + FEATURE_ORDER.length + " values per instance.");
        }
        Map<String, Object> values = new LinkedHashMap<>();
        for (int i = 0; i < FEATURE_ORDER.length; i++) {
            String feature = FEATURE_ORDER[i];
            String token = tokens[i];
            if (NUMERIC_FEATURES.contains(feature)) {
                try {
                    values.put(feature, Double.parseDouble(token));
                } catch (NumberFormatException nfe) {
                    throw new IllegalArgumentException("Cannot parse numeric value for " + feature + ": " + token, nfe);
                }
            } else {
                values.put(feature, token);
            }
        }
        // Categorical features expect strings
        values.putIfAbsent("Sex", "male");
        values.putIfAbsent("Embarked", "S");
        return values;
    }

    private static List<Map<String, Object>> readBatchFile(Path batchPath) {
        Path candidate = batchPath;
        if (!Files.exists(candidate)) {
            Path parentCandidate = Path.of("..").resolve(batchPath);
            if (Files.exists(parentCandidate)) {
                candidate = parentCandidate;
            }
        }
        if (!Files.exists(candidate)) {
            throw new IllegalArgumentException("Batch file not found: " + batchPath);
        }
        List<Map<String, Object>> vectors = new ArrayList<>();
        try {
            List<String> lines = Files.readAllLines(candidate);
            for (String line : lines) {
                String trimmed = line.trim();
                if (trimmed.isEmpty() || trimmed.startsWith("#")) {
                    continue;
                }
                vectors.add(parseFeatureVector(trimmed.split("[,\\s]+")));
            }
        } catch (IOException ex) {
            throw new IllegalStateException("Failed to read batch file: " + batchPath, ex);
        }
        return vectors;
    }

    private static Map<FieldName, FieldValue> prepareArguments(
        Evaluator evaluator,
        Map<String, Object> featureValues
    ) {
        Map<FieldName, FieldValue> arguments = new LinkedHashMap<>();
        for (InputField inputField : evaluator.getInputFields()) {
            FieldName fieldName = inputField.getName();
            Object rawValue = featureValues.get(fieldName.getValue());
            if (rawValue == null) {
                throw new IllegalArgumentException("Missing value for required field: " + fieldName.getValue());
            }
            arguments.put(fieldName, inputField.prepare(rawValue));
        }
        return arguments;
    }

    private static void printInput(Map<String, Object> features) {
        System.out.println("Input features (Titanic order: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):");
        for (String feature : FEATURE_ORDER) {
            Object value = features.get(feature);
            if (value instanceof Number number) {
                System.out.printf(Locale.US, "  %-9s = %.4f%n", feature, number.doubleValue());
            } else {
                System.out.printf(Locale.US, "  %-9s = %s%n", feature, value);
            }
        }
        System.out.println();
    }

    private static void printResults(Map<FieldName, ?> results, Evaluator evaluator) {
        TargetField targetField = evaluator.getTargetFields().get(0);
        FieldName targetFieldName = targetField.getName();
        Object predictedValue = unwrapComputable(results.get(targetFieldName));
        System.out.printf("Predicted class id: %s%n", predictedValue);
        System.out.printf("Predicted class label: %s%n", CLASS_LABELS.getOrDefault(String.valueOf(predictedValue), "unknown"));
        System.out.println();
        System.out.println("Class probabilities:");
        printProbabilities(results, evaluator.getOutputFields());
    }

    private static void printProbabilities(Map<FieldName, ?> results, List<OutputField> outputFields) {
        DecimalFormat formatter = new DecimalFormat("0.0000", DecimalFormatSymbols.getInstance(Locale.US));
        for (OutputField outputField : outputFields) {
            FieldName name = outputField.getName();
            Object computedValue = unwrapComputable(results.get(name));
            if (computedValue instanceof Number number) {
                String formatted = formatter.format(number.doubleValue());
                System.out.printf(Locale.US, "  %-16s : %s%n", name.getValue(), formatted);
            } else if (computedValue instanceof Map<?, ?> map) {
                for (Map.Entry<?, ?> entry : map.entrySet()) {
                    Object value = entry.getValue();
                    if (value instanceof Number numberEntry) {
                        String formatted = formatter.format(numberEntry.doubleValue());
                        System.out.printf(Locale.US, "  %-16s : %s%n", entry.getKey(), formatted);
                    }
                }
            } else if (computedValue != null) {
                System.out.printf(Locale.US, "  %-16s : %s%n", name.getValue(), computedValue);
            }
        }
    }

    private static Object unwrapComputable(Object value) {
        if (value instanceof Computable computable) {
            return computable.getResult();
        }
        return value;
    }

    private record ParsedInput(Path modelPath, List<Map<String, Object>> featureVectors, boolean modelExplicit) {
    }
}
