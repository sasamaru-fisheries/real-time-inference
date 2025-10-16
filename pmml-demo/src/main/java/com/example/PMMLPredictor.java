package com.example;

import java.io.IOException;
import java.io.InputStream;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Arrays;
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
        Evaluator evaluator = loadEvaluator();
        Map<String, Object> featureValues = parseArguments(args);

        Map<FieldName, FieldValue> arguments = prepareArguments(evaluator, featureValues);
        Map<FieldName, ?> results = evaluator.evaluate(arguments);

        TargetField targetField = evaluator.getTargetFields().get(0);
        FieldName targetFieldName = targetField.getName();
        Object predictedValue = unwrapComputable(results.get(targetFieldName));

        System.out.println("Input features (order: sepal_length, sepal_width, petal_length, petal_width):");
        Arrays.stream(FEATURE_ORDER).forEach(name ->
            System.out.printf(Locale.US, "  %-22s = %.2f%n", name, ((Number) featureValues.get(name)).doubleValue())
        );
        System.out.println();

        System.out.printf("Predicted class id: %s%n", predictedValue);
        System.out.printf("Predicted class label: %s%n", CLASS_LABELS.getOrDefault(String.valueOf(predictedValue), "unknown"));

        System.out.println();
        System.out.println("Class probabilities:");
        printProbabilities(results, evaluator.getOutputFields());
    }

    private static Evaluator loadEvaluator() throws Exception {
        PMML pmml;
        try (InputStream is = PMMLPredictor.class.getResourceAsStream("/model.pmml")) {
            if (is == null) {
                throw new IOException("Could not find model.pmml on the classpath.");
            }
            pmml = PMMLUtil.unmarshal(is);
        }

        Evaluator evaluator = new ModelEvaluatorBuilder(pmml)
            .build();
        evaluator.verify();
        return evaluator;
    }

    private static Map<String, Object> parseArguments(String[] args) {
        Map<String, Object> values = new LinkedHashMap<>();

        if (args.length != 0 && args.length != FEATURE_ORDER.length) {
            throw new IllegalArgumentException(
                "Provide exactly four values (sepal_length sepal_width petal_length petal_width) or no values to use defaults.");
        }

        if (args.length == 0) {
            values.put("sepal length (cm)", 5.1d);
            values.put("sepal width (cm)", 3.5d);
            values.put("petal length (cm)", 1.4d);
            values.put("petal width (cm)", 0.2d);
            return values;
        }

        for (int i = 0; i < FEATURE_ORDER.length; i++) {
            double parsedValue = Double.parseDouble(args[i]);
            values.put(FEATURE_ORDER[i], parsedValue);
        }
        return values;
    }

    private static Map<FieldName, FieldValue> prepareArguments(Evaluator evaluator, Map<String, Object> featureValues) {
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
        if (value instanceof Computable) {
            return ((Computable) value).getResult();
        }
        return value;
    }
}
