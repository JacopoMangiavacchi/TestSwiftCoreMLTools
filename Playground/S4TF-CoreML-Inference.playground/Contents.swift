import Foundation
import CoreML

func compileCoreML(path: String) -> (MLModel, URL) {
    let modelUrl = URL(fileURLWithPath: path)
    let compiledUrl = try! MLModel.compileModel(at: modelUrl)
    
    print("Compiled Model Path: \(compiledUrl)")
    return try! (MLModel(contentsOf: compiledUrl), compiledUrl)
}

func inferenceCoreML(model: MLModel, x: [Float]) -> Float {
    let inputName = "input"
    
    let multiArr = try! MLMultiArray(shape: [13], dataType: .double)
    for i in 0..<13 {
        multiArr[i] = NSNumber(value: x[i])
    }

    let inputValue = MLFeatureValue(multiArray: multiArr)
    let dataPointFeatures: [String: MLFeatureValue] = [inputName: inputValue]
    let provider = try! MLDictionaryFeatureProvider(dictionary: dataPointFeatures)
    
    let prediction = try! model.prediction(from: provider)

    return Float(prediction.featureValue(for: "output")!.multiArrayValue![0].doubleValue)
}

let (coreModel, compiledModelUrl) = compileCoreML(path: "/Users/jacopo/TestSwiftCoreMLTools/model/s4tf_train_model.mlmodel")

// print(coreModel.modelDescription)

let record: [Float] = [-0.30846816,   0.15054052,   -1.1039964,  -0.30756173,   0.05126577,   0.33352128,
0.024851447, -0.035726987,  -0.89119065,  -0.43651816,   -1.2319369,   0.42111015,
-0.93603975] // Expected 24.0 !!!

let prediction = inferenceCoreML(model: coreModel, x: record)
print(prediction)
