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

let (coreModel, compiledModelUrl) = compileCoreML(path: "/Users/jacopo/TestCoreMLTools/model/coreml_model_double_train.mlmodel")

// print(coreModel.modelDescription)

let record: [Float] = [-0.38987719,  0.06312032, -0.48184431, -0.27218959, -0.25178915, -0.38486309,
                       0.63271075,  1.27556958, -0.51912059, -0.5780854,  -1.51173314,  0.33334089, 0.62918333] // Expected 18.9


let prediction = inferenceCoreML(model: coreModel, x: record)
print(prediction)
