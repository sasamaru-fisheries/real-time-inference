package com.example;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Map.Entry;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.RunOptions;

/**
 * Command line ONNX inference utility compatible with the exported Iris classifier.
 *
 * Usage examples:
 *   java -cp target/pmml-demo-1.0-SNAPSHOT.jar com.example.OnnxPredictor --model model/model.onnx
 *   java -cp target/pmml-demo-1.0-SNAPSHOT.jar com.example.OnnxPredictor --model model/model.onnx 6.1 2.8 4.7 1.2
 */
public final class OnnxPredictor {

    private static final String MODEL_OPTION = "--model";
    private static final Path DEFAULT_MODEL_PATH = Path.of("model", "model.onnx");

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

    private OnnxPredictor() {
        // Utility class; do not instantiate.
    }

    public static void main(String[] args) throws Exception {
        ParsedInput parsedInput = parseInput(args);
        Path resolvedModelPath = resolveModelPath(parsedInput.modelPath(), parsedInput.modelExplicit());

        float[] featureValues = parsedInput.featureValues();

        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
             OrtSession.SessionOptions options = new OrtSession.SessionOptions();
             OrtSession session = env.createSession(resolvedModelPath.toString(), options)) {

            String inputName = session.getInputNames().iterator().next();
            float[][] inputData = new float[][] { featureValues };

            try (OnnxTensor tensor = OnnxTensor.createTensor(env, inputData);
                 RunOptions runOptions = new RunOptions();
                 OrtSession.Result result = session.run(Map.of(inputName, tensor), runOptions)) {

                PredictionOutputs outputs = extractOutputs(result);
                long predictedClass = outputs.predictedClass();

                System.out.println("Input features (order: sepal_length, sepal_width, petal_length, petal_width):");
                for (int i = 0; i < FEATURE_ORDER.length; i++) {
                    System.out.printf(Locale.US, "  %-22s = %.2f%n", FEATURE_ORDER[i], featureValues[i]);
                }

                System.out.println();
                System.out.printf("Predicted class id: %d%n", predictedClass);
                System.out.printf("Predicted class label: %s%n", CLASS_LABELS.getOrDefault(String.valueOf(predictedClass), "unknown"));
                System.out.println();
                System.out.println("Model loaded from: " + resolvedModelPath.toAbsolutePath());
                System.out.println();
                System.out.println("Class probabilities:");
                for (int i = 0; i < outputs.probabilities().length; i++) {
                    System.out.printf(Locale.US, "  %-16s : %.4f%n", "probability(" + i + ")", outputs.probabilities()[i]);
                }
            }
        }
    }

    private static ParsedInput parseInput(String[] args) {
        Path modelPath = DEFAULT_MODEL_PATH;
        boolean explicitModelPath = false;
        List<String> featureArgs = new ArrayList<>();

        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            if (MODEL_OPTION.equals(arg)) {
                if ((i + 1) >= args.length) {
                    throw new IllegalArgumentException("--model requires a file path argument.");
                }
                modelPath = Path.of(args[++i]);
                explicitModelPath = true;
            } else if (arg.startsWith(MODEL_OPTION + "=")) {
                modelPath = Path.of(arg.substring((MODEL_OPTION + "=").length()));
                explicitModelPath = true;
            } else {
                featureArgs.add(arg);
            }
        }

        float[] featureValues = parseFeatures(featureArgs.toArray(String[]::new));
        return new ParsedInput(modelPath, featureValues, explicitModelPath);
    }

    private static float[] parseFeatures(String[] args) {
        if (args.length != 0 && args.length != FEATURE_ORDER.length) {
            throw new IllegalArgumentException(
                "Provide exactly four values (sepal_length sepal_width petal_length petal_width) or no values to use defaults.");
        }

        float[] values = new float[FEATURE_ORDER.length];
        if (args.length == 0) {
            values[0] = 5.1f;
            values[1] = 3.5f;
            values[2] = 1.4f;
            values[3] = 0.2f;
            return values;
        }

        for (int i = 0; i < FEATURE_ORDER.length; i++) {
            values[i] = Float.parseFloat(args[i]);
        }
        return values;
    }

    private static Path resolveModelPath(Path modelPath, boolean explicitModelPath) throws IOException {
        if (Files.exists(modelPath)) {
            return modelPath;
        }

        if (explicitModelPath || modelPath.isAbsolute()) {
            throw new IOException("Could not find ONNX model file at: " + modelPath.toAbsolutePath());
        }

        Path parentCandidate = Path.of("..").resolve(modelPath);
        if (Files.exists(parentCandidate)) {
            return parentCandidate;
        }

        throw new IOException(
            "Could not find ONNX model file. Checked: " + modelPath.toAbsolutePath() + " and " + parentCandidate.toAbsolutePath());
    }

    private static PredictionOutputs extractOutputs(OrtSession.Result result) throws OrtException {
        Long predictedClass = null;
        double[] probabilities = null;

        for (Entry<String, OnnxValue> entry : result) {
            String name = entry.getKey();
            Object value = entry.getValue().getValue();

            if (predictedClass == null && isLabelOutput(name)) {
                predictedClass = extractLabel(value);
            } else if (probabilities == null && isProbabilityOutput(name)) {
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

    private static boolean isLabelOutput(String outputName) {
        return outputName != null && outputName.toLowerCase(Locale.ROOT).contains("label");
    }

    private static boolean isProbabilityOutput(String outputName) {
        return outputName != null && (outputName.toLowerCase(Locale.ROOT).contains("prob") || outputName.toLowerCase(Locale.ROOT).contains("score"));
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
            try {
                return Long.parseLong(label);
            } catch (NumberFormatException ex) {
                throw new IllegalStateException("Unable to interpret label output: " + label, ex);
            }
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
            double[] probs = new double[map.size()];
            int index = 0;
            for (Entry<?, ?> entry : map.entrySet()) {
                Object value = entry.getValue();
                if (value instanceof Number number) {
                    probs[index++] = number.doubleValue();
                }
            }
            return probs;
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

    private record ParsedInput(Path modelPath, float[] featureValues, boolean modelExplicit) { }

    private record PredictionOutputs(long predictedClass, double[] probabilities) { }
}
