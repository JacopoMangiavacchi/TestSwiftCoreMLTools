# TestSwiftCoreMLTools
Test [SwiftCoreMLTools](https://github.com/JacopoMangiavacchi/SwiftCoreMLTools) library exporting to CoreML a real Swift for TensorFlow Regression model trained on the [Boston housing price dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/).

This repo contains:

- The S4TF training notebook exporting to CoreML with the SwiftCoreMLTools swift library
- A TensorfFlow 1.x training notebook exporting to CoreML with the CoreMLTools python library
- A TensorfFlow 2.x training notebook exporting to CoreML with the tfcoreml python library
- Xcode playgrounds to test inferencing the exported CoreML models

## Swift for TensorFlow Model

```swift
struct RegressionModel: Layer {
    var layer1 = Dense<Float>(inputSize: 13, outputSize: 64, activation: relu)
    var layer2 = Dense<Float>(inputSize: 64, outputSize: 32, activation: relu)
    var layer3 = Dense<Float>(inputSize: 32, outputSize: 1)
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: layer1, layer2, layer3)
    }
}

var model = RegressionModel()
...
// Training
...
```

## CoreML Model Export with SwiftCoreMLTools

```swift
let coremlModel = Model(version: 4,
                        shortDescription: "Regression",
                        author: "Jacopo Mangiavacchi",
                        license: "MIT",
                        userDefined: ["SwiftCoremltoolsVersion" : "0.0.3"]) {
    Input(name: "input", shape: [13])
    Output(name: "output", shape: [1])
    NeuralNetwork {
        InnerProduct(name: "dense1",
                     input: ["input"],
                     output: ["outDense1"],
                     weight: model.layer1.weight.transposed().flattened().scalars,
                     bias: model.layer1.bias.flattened().scalars,
                     inputChannels: 13,
                     outputChannels: 64)
        ReLu(name: "Relu1",
             input: ["outDense1"],
             output: ["outRelu1"])
        InnerProduct(name: "dense2",
                     input: ["outRelu1"],
                     output: ["outDense2"],
                     weight: model.layer2.weight.transposed().flattened().scalars,
                     bias: model.layer2.bias.flattened().scalars,
                     inputChannels: 64,
                     outputChannels: 32)
        ReLu(name: "Relu2",
             input: ["outDense2"],
             output: ["outRelu2"])
        InnerProduct(name: "dense3",
                     input: ["outRelu2"],
                     output: ["output"],
                     weight: model.layer3.weight.transposed().flattened().scalars,
                     bias: model.layer3.bias.flattened().scalars,
                     inputChannels: 32,
                     outputChannels: 1)
    }
}
```
