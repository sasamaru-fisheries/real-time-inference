This directory must contain `onnxruntime-platform-1.19.2.jar` before you build the Maven module.

Fetch the jar with:

```bash
mvn -q dependency:copy -Dartifact=com.microsoft.onnxruntime:onnxruntime:1.19.2 -DoutputDirectory=onnx-predictor/libs
mv onnx-predictor/libs/onnxruntime-1.19.2.jar onnx-predictor/libs/onnxruntime-platform-1.19.2.jar
```

`mvn clean package` in `onnx-predictor/` will then bundle it into the shaded CLI.
