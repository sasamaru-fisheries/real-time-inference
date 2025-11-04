package com.example;

import ai.onnxruntime.OnnxMap;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.RunOptions;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

/**
 * CLI for scoring ONNX models (デフォルト: Titanic RandomForest {@code model/titanic_random_forest.onnx}).
 *
 * <p>Usage examples:
 * <pre>
 *   mvn -q -pl onnx-predictor exec:java -Dexec.args="3 male 22 1 0 7.25 S"
 *   mvn -q -pl onnx-predictor exec:java -Dexec.args="--batch data/sample_batch.txt"
 *   java -jar target/onnx-predictor-1.0-SNAPSHOT.jar --model other.onnx 3 male 22 1 0 7.25 S
 * </pre>
 */
public final class OnnxPredictor {

    private static final String MODEL_OPTION = "--model";
    private static final String BATCH_OPTION = "--batch";
    private static final Path DEFAULT_MODEL_PATH = Path.of("model", "titanic_random_forest.onnx");

    private static final String[] FEATURE_NAMES = {
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked"
    };

    private static final Set<String> STRING_FEATURES = new LinkedHashSet<>(List.of(
        "Pclass", "Sex", "Embarked"
    ));

    private static final Set<String> FLOAT_FEATURES = new LinkedHashSet<>(List.of(
        "Age", "SibSp", "Parch", "Fare"
    ));

    private static final Map<String, String> CLASS_LABELS = Map.of(
        "0", "not_survived",
        "1", "survived"
    );

    private OnnxPredictor() {
    }

    public static void main(String[] args) throws Exception {
        ParsedInput parsed = parseInput(args);
        Path modelPath = resolveModelPath(parsed.modelPath(), parsed.modelExplicit());
        List<Map<String, Object>> samples = parsed.samples();

        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             OrtSession session = env.createSession(modelPath.toString(), new OrtSession.SessionOptions())) {

            for (int i = 0; i < samples.size(); i++) {
                Map<String, Object> features = samples.get(i);
                Map<String, OnnxTensor> inputs = buildInputTensors(env, session, features);

                try (RunOptions options = new RunOptions();
                     OrtSession.Result result = session.run(inputs, options)) {

                    PredictionOutputs outputs = extractOutputs(result);
                    if (samples.size() > 1) {
                        System.out.printf("=== Sample %d ===%n", i + 1);
                    }
                    System.out.println("Input features (order: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):");
                    for (String feature : FEATURE_NAMES) {
                        Object value = features.get(feature);
                        if (value instanceof Number number) {
                            System.out.printf(Locale.US, "  %-12s = %.2f%n", feature, number.doubleValue());
                        } else {
                            System.out.printf("  %-12s = %s%n", feature, value);
                        }
                    }
                    System.out.println();
                    System.out.printf("Predicted class id: %d%n", outputs.predictedClass());
                    System.out.printf("Predicted class label: %s%n",
                        CLASS_LABELS.getOrDefault(String.valueOf(outputs.predictedClass()), "unknown"));
                    System.out.println("Model loaded from: " + modelPath.toAbsolutePath());
                    System.out.println("Class probabilities:");
                    for (int j = 0; j < outputs.probabilities().length; j++) {
                        System.out.printf(Locale.US, "  probability(%d)   : %.4f%n", j, outputs.probabilities()[j]);
                    }
                    if (i < samples.size() - 1) {
                        System.out.println();
                    }
                } finally {
                    closeInputs(inputs);
                }
            }
        }
    }

    private static ParsedInput parseInput(String[] args) {
        Path modelPath = DEFAULT_MODEL_PATH;
        boolean explicitModel = false;
        Path batchPath = null;
        List<String> featureArgs = new ArrayList<>();

        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            if (MODEL_OPTION.equals(arg)) {
                if ((i + 1) >= args.length) {
                    throw new IllegalArgumentException("--model requires a file path.");
                }
                modelPath = Path.of(args[++i]);
                explicitModel = true;
            } else if (arg.startsWith(MODEL_OPTION + "=")) {
                modelPath = Path.of(arg.substring((MODEL_OPTION + "=").length()));
                explicitModel = true;
            } else if (BATCH_OPTION.equals(arg)) {
                if ((i + 1) >= args.length) {
                    throw new IllegalArgumentException("--batch requires a file path.");
                }
                batchPath = Path.of(args[++i]);
            } else if (arg.startsWith(BATCH_OPTION + "=")) {
                batchPath = Path.of(arg.substring((BATCH_OPTION + "=").length()));
            } else {
                featureArgs.add(arg);
            }
        }

        List<Map<String, Object>> samples = new ArrayList<>();
        if (!featureArgs.isEmpty()) {
            samples.add(parseFeatures(featureArgs.toArray(String[]::new)));
        }
        if (batchPath != null) {
            samples.addAll(readBatchFile(batchPath));
        }
        if (samples.isEmpty()) {
            samples.add(parseFeatures(new String[] {"3", "male", "22", "1", "0", "7.25", "S"}));
        }
        return new ParsedInput(modelPath, samples, explicitModel);
    }

    private static Map<String, Object> parseFeatures(String[] args) {
        if (args.length != FEATURE_NAMES.length) {
            throw new IllegalArgumentException("Expected seven values per sample.");
        }

        Map<String, Object> values = new LinkedHashMap<>();
        for (int i = 0; i < FEATURE_NAMES.length; i++) {
            String feature = FEATURE_NAMES[i];
            String raw = args[i];
            if (STRING_FEATURES.contains(feature)) {
                values.put(feature, raw);
            } else if (FLOAT_FEATURES.contains(feature)) {
                try {
                    values.put(feature, Float.parseFloat(raw));
                } catch (NumberFormatException ex) {
                    throw new IllegalArgumentException(
                        "Expected numeric value for feature '" + feature + "', but got: " + raw, ex);
                }
            } else {
                values.put(feature, raw);
            }
        }
        return values;
    }

    private static Path resolveModelPath(Path modelPath, boolean explicit) throws IOException {
        if (Files.exists(modelPath)) {
            return modelPath;
        }
        if (explicit || modelPath.isAbsolute()) {
            throw new IOException("Could not find ONNX model at " + modelPath.toAbsolutePath());
        }
        Path parentCandidate = Path.of("..").resolve(modelPath);
        if (Files.exists(parentCandidate)) {
            return parentCandidate;
        }
        throw new IOException("Could not find ONNX model. Checked " + modelPath.toAbsolutePath()
            + " and " + parentCandidate.toAbsolutePath());
    }

    private static List<Map<String, Object>> readBatchFile(Path batchPath) {
        Path resolved = batchPath;
        if (!Files.exists(resolved)) {
            Path parentCandidate = Path.of("..").resolve(batchPath);
            if (Files.exists(parentCandidate)) {
                resolved = parentCandidate;
            }
        }
        if (!Files.exists(resolved)) {
            throw new IllegalArgumentException("Batch file not found: " + batchPath);
        }

        List<Map<String, Object>> samples = new ArrayList<>();
        try {
            for (String line : Files.readAllLines(resolved)) {
                String trimmed = line.trim();
                if (trimmed.isEmpty() || trimmed.startsWith("#")) {
                    continue;
                }
                String[] parts = trimmed.split("[,\\s]+");
                samples.add(parseFeatures(parts));
            }
        } catch (IOException ex) {
            throw new IllegalStateException("Failed to read batch file: " + batchPath, ex);
        }
        return samples;
    }

    private static Map<String, OnnxTensor> buildInputTensors(
        OrtEnvironment env,
        OrtSession session,
        Map<String, Object> features
    ) throws OrtException {
        Map<String, OnnxTensor> tensors = new LinkedHashMap<>();
        for (String inputName : session.getInputNames()) {
            Object value = features.get(inputName);
            if (value == null) {
                throw new IllegalArgumentException("Missing feature '" + inputName + "' in input sample.");
            }

            if (STRING_FEATURES.contains(inputName)) {
                String[][] data = new String[][] { { value.toString() } };
                tensors.put(inputName, OnnxTensor.createTensor(env, data));
            } else if (FLOAT_FEATURES.contains(inputName)) {
                float floatValue;
                if (value instanceof Number number) {
                    floatValue = number.floatValue();
                } else {
                    floatValue = Float.parseFloat(value.toString());
                }
                float[][] data = new float[][] { { floatValue } };
                tensors.put(inputName, OnnxTensor.createTensor(env, data));
            } else {
                throw new IllegalArgumentException("Unhandled feature type for '" + inputName + "'.");
            }
        }
        return tensors;
    }

    private static PredictionOutputs extractOutputs(OrtSession.Result result) throws OrtException {
        Long predictedClass = null;
        double[] probabilities = null;

        for (Entry<String, OnnxValue> entry : result) {
            Object value = entry.getValue().getValue();
            String name = entry.getKey().toLowerCase(Locale.ROOT);

            if (predictedClass == null && name.contains("label")) {
                predictedClass = extractLabel(value);
            } else if (probabilities == null && (name.contains("prob") || name.contains("score"))) {
                probabilities = extractProbabilities(value);
            }
        }

        if (predictedClass == null) {
            throw new IllegalStateException("ONNX output did not contain a label tensor.");
        }
        if (probabilities == null) {
            probabilities = new double[0];
        }
        return new PredictionOutputs(predictedClass, probabilities);
    }

    private static long extractLabel(Object rawLabel) {
        if (rawLabel instanceof long[] array && array.length > 0) {
            return array[0];
        }
        if (rawLabel instanceof int[] array && array.length > 0) {
            return array[0];
        }
        if (rawLabel instanceof float[] array && array.length > 0) {
            return Math.round(array[0]);
        }
        if (rawLabel instanceof double[] array && array.length > 0) {
            return Math.round(array[0]);
        }
        if (rawLabel instanceof String[] array && array.length > 0) {
            String label = array[0];
            for (Entry<String, String> entry : CLASS_LABELS.entrySet()) {
                if (entry.getValue().equalsIgnoreCase(label)) {
                    return Long.parseLong(entry.getKey());
                }
            }
            return Long.parseLong(label);
        }
        throw new IllegalStateException("Unsupported label tensor type: " + rawLabel.getClass());
    }

    private static double[] extractProbabilities(Object rawProbabilities) {
        if (rawProbabilities instanceof float[][] array && array.length > 0) {
            return convertToDouble(array[0]);
        }
        if (rawProbabilities instanceof double[][] array && array.length > 0) {
            return array[0];
        }
        if (rawProbabilities instanceof float[] array) {
            return convertToDouble(array);
        }
        if (rawProbabilities instanceof double[] array) {
            return array;
        }
        if (rawProbabilities instanceof Map<?, ?> map) {
            return convertMapToArray(map);
        }
        if (rawProbabilities instanceof List<?> list) {
            if (list.isEmpty()) {
                return new double[0];
            }

            Object first = list.get(0);
            if (first instanceof Number) {
                double[] probs = new double[list.size()];
                for (int i = 0; i < list.size(); i++) {
                    Object value = list.get(i);
                    if (value instanceof Number number) {
                        probs[i] = number.doubleValue();
                    } else {
                        throw new IllegalStateException("Mixed probability list types: " + value.getClass());
                    }
                }
                return probs;
            }

            if (first instanceof OnnxMap onnxMap) {
                try {
                    return convertMapToArray(onnxMap.getValue());
                } catch (OrtException ex) {
                    throw new IllegalStateException("Failed to extract probabilities from OnnxMap", ex);
                }
            }

            if (first instanceof Map<?, ?> nestedMap) {
                return convertMapToArray(nestedMap);
            }

            throw new IllegalStateException("Unsupported list element type in probabilities: " + first.getClass());
        }
        throw new IllegalStateException("Unsupported probability tensor type: " + rawProbabilities.getClass());
    }

    private static double[] convertToDouble(float[] array) {
        double[] result = new double[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = array[i];
        }
        return result;
    }

    private static void closeInputs(Map<String, OnnxTensor> inputs) {
        for (OnnxTensor tensor : inputs.values()) {
            try {
                tensor.close();
            } catch (RuntimeException ignored) {
                // クローズ時の例外は無視する
            }
        }
    }

    private static double[] convertMapToArray(Map<?, ?> map) {
        List<Entry<?, ?>> entries = new ArrayList<>(map.entrySet());
        entries.sort(Comparator.comparing(entry -> entry.getKey().toString()));
        double[] probs = new double[entries.size()];
        for (int i = 0; i < entries.size(); i++) {
            Object value = entries.get(i).getValue();
            if (value instanceof Number number) {
                probs[i] = number.doubleValue();
            } else {
                throw new IllegalStateException("Unsupported probability map value type: " + value.getClass());
            }
        }
        return probs;
    }

    private record ParsedInput(Path modelPath, List<Map<String, Object>> samples, boolean modelExplicit) { }

    private record PredictionOutputs(long predictedClass, double[] probabilities) { }
}
