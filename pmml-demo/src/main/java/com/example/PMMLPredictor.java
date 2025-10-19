package com.example;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

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
 * CLI tool to perform inference using the bundled PMML model.
 * Usage:
 *   mvn -q exec:java -Dexec.args="5.1 3.5 1.4 0.2"
 * If no arguments are supplied, default Iris setosa measurements are used.
 */
public final class PMMLPredictor {

    private static final String MODEL_OPTION = "--model";
    private static final String BATCH_OPTION = "--batch";
    private static final Path DEFAULT_MODEL_PATH = Path.of("model", "model.pmml");

    private static final String[] FEATURE_ORDER = {
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    };

    private static final Map<String, String> CLASS_LABELS = Map.of(
        "0", "setosa",
        "1", "versicolor",
        "2", "virginica"
    );

    private PMMLPredictor() {
        // Utility class; do not instantiate.
    }

    public static void main(String[] args) throws Exception {
        ParsedInput parsedInput = parseInput(args);
        Path resolvedModelPath = resolveModelPath(parsedInput.modelPath(), parsedInput.modelExplicit());

        Evaluator evaluator = loadEvaluator(resolvedModelPath);
        List<double[]> featureVectors = parsedInput.featureVectors();

        System.out.println("Model loaded from: " + resolvedModelPath.toAbsolutePath());
        System.out.println();

        for (int i = 0; i < featureVectors.size(); i++) {
            double[] features = featureVectors.get(i);
            Map<FieldName, FieldValue> arguments = prepareArguments(evaluator, features);
            Map<FieldName, ?> results = evaluator.evaluate(arguments);

            TargetField targetField = evaluator.getTargetFields().get(0);
            FieldName targetFieldName = targetField.getName();
            Object predictedValue = unwrapComputable(results.get(targetFieldName));

            if (featureVectors.size() > 1) {
                System.out.printf("=== Sample %d ===%n", i + 1);
            }

            System.out.println("Input features (order: sepal_length, sepal_width, petal_length, petal_width):");
            for (int j = 0; j < FEATURE_ORDER.length; j++) {
                System.out.printf(Locale.US, "  %-22s = %.2f%n", FEATURE_ORDER[j], features[j]);
            }
            System.out.println();

            System.out.printf("Predicted class id: %s%n", predictedValue);
            System.out.printf("Predicted class label: %s%n", CLASS_LABELS.getOrDefault(String.valueOf(predictedValue), "unknown"));
            System.out.println();
            System.out.println("Class probabilities:");
            printProbabilities(results, evaluator.getOutputFields());

            if (i < featureVectors.size() - 1) {
                System.out.println();
            }
        }
    }

    private static ParsedInput parseInput(String[] args) {
        Path modelPath = DEFAULT_MODEL_PATH;
        boolean explicitModelPath = false;
        List<String> featureArgs = new ArrayList<>();
        Path batchPath = null;

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
            } else if (arg.startsWith(MODEL_OPTION + "=")) {
                modelPath = Path.of(arg.substring((MODEL_OPTION + "=").length()));
                explicitModelPath = true;
            } else if (arg.startsWith(BATCH_OPTION + "=")) {
                batchPath = Path.of(arg.substring((BATCH_OPTION + "=").length()));
            } else {
                featureArgs.add(arg);
            }
        }

        List<double[]> featureVectors = new ArrayList<>();

        if (!featureArgs.isEmpty()) {
            featureVectors.add(parseFeatureVector(featureArgs.toArray(String[]::new)));
        }

        if (batchPath != null) {
            featureVectors.addAll(readBatchFile(batchPath));
        }

        if (featureVectors.isEmpty()) {
            featureVectors.add(new double[] {5.1d, 3.5d, 1.4d, 0.2d});
        }

        return new ParsedInput(modelPath, featureVectors, explicitModelPath);
    }

    private static Path resolveModelPath(Path modelPath, boolean explicitModelPath) throws IOException {
        if (Files.exists(modelPath)) {
            return modelPath;
        }

        if (explicitModelPath || modelPath.isAbsolute()) {
            throw new IOException("Could not find PMML model file at: " + modelPath.toAbsolutePath());
        }

        Path parentCandidate = Path.of("..").resolve(modelPath);
        if (Files.exists(parentCandidate)) {
            return parentCandidate;
        }

        throw new IOException(
            "Could not find PMML model file. Checked: " + modelPath.toAbsolutePath() + " and " + parentCandidate.toAbsolutePath());
    }

    private static Evaluator loadEvaluator(Path resolvedModelPath) throws Exception {

        PMML pmml;
        try (InputStream is = Files.newInputStream(resolvedModelPath)) {
            pmml = PMMLUtil.unmarshal(is);
        }

        Evaluator evaluator = new ModelEvaluatorBuilder(pmml)
            .build();
        evaluator.verify();
        return evaluator;
    }

    private static Map<FieldName, FieldValue> prepareArguments(Evaluator evaluator, double[] featureValues) {
        Map<String, Object> featureMap = new LinkedHashMap<>();
        for (int i = 0; i < FEATURE_ORDER.length; i++) {
            featureMap.put(FEATURE_ORDER[i], featureValues[i]);
        }

        Map<FieldName, FieldValue> arguments = new LinkedHashMap<>();
        for (InputField inputField : evaluator.getInputFields()) {
            FieldName fieldName = inputField.getName();
            Object rawValue = featureMap.get(fieldName.getValue());
            if (rawValue == null) {
                throw new IllegalArgumentException("Missing value for required field: " + fieldName.getValue());
            }
            arguments.put(fieldName, inputField.prepare(rawValue));
        }
        return arguments;
    }

    private static void printProbabilities(Map<FieldName, ?> results, List<OutputField> outputFields) {
        DecimalFormat formatter = new DecimalFormat("0.0000", DecimalFormatSymbols.getInstance(Locale.US));

        for (OutputField outputField : outputFields) {
            FieldName name = outputField.getName();
            Object computedValue = unwrapComputable(results.get(name));

            if (computedValue instanceof Number) {
                String formatted = formatter.format(((Number) computedValue).doubleValue());
                System.out.printf(Locale.US, "  %-16s : %s%n", name.getValue(), formatted);
            } else if (computedValue instanceof Map<?, ?> map) {
                for (Map.Entry<?, ?> entry : map.entrySet()) {
                    Object entryValue = entry.getValue();
                    if (entryValue instanceof Number) {
                        String formatted = formatter.format(((Number) entryValue).doubleValue());
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

    private static double[] parseFeatureVector(String[] args) {
        if (args.length != FEATURE_ORDER.length) {
            throw new IllegalArgumentException(
                "Provide exactly four values (sepal_length sepal_width petal_length petal_width) or use --batch for multiple samples.");
        }
        double[] vector = new double[FEATURE_ORDER.length];
        for (int i = 0; i < FEATURE_ORDER.length; i++) {
            vector[i] = Double.parseDouble(args[i]);
        }
        return vector;
    }

    private static List<double[]> readBatchFile(Path batchPath) {
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

        List<double[]> vectors = new ArrayList<>();
        try {
            List<String> lines = Files.readAllLines(candidate);
            for (String line : lines) {
                String trimmed = line.trim();
                if (trimmed.isEmpty() || trimmed.startsWith("#")) {
                    continue;
                }
                String[] parts = trimmed.split("[,\\s]+");
                if (parts.length != FEATURE_ORDER.length) {
                    throw new IllegalArgumentException("Invalid line in batch file (expected 4 values): " + line);
                }
                vectors.add(parseFeatureVector(parts));
            }
        } catch (IOException ex) {
            throw new IllegalStateException("Failed to read batch file: " + batchPath, ex);
        }
        return vectors;
    }

    private record ParsedInput(Path modelPath, List<double[]> featureVectors, boolean modelExplicit) { }
}
