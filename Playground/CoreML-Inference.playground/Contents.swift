import Foundation
import CoreML

func compileCoreML(path: String) -> (MLModel, URL) {
    let modelUrl = URL(fileURLWithPath: path)
    let compiledUrl = try! MLModel.compileModel(at: modelUrl)
    
    print("Compiled Model Path: \(compiledUrl)")
    return try! (MLModel(contentsOf: compiledUrl), compiledUrl)
}

func inferenceCoreML(model: MLModel, x: [Float]) -> Float {
    let inputName = "input1"
    
    let multiArr = try! MLMultiArray(shape: [13], dataType: .double)
    for i in 0..<13 {
        multiArr[i] = NSNumber(value: x[i])
    }

    let inputValue = MLFeatureValue(multiArray: multiArr)
    let dataPointFeatures: [String: MLFeatureValue] = [inputName: inputValue]
    let provider = try! MLDictionaryFeatureProvider(dictionary: dataPointFeatures)
    
    let prediction = try! model.prediction(from: provider)

    return Float(prediction.featureValue(for: "output1")!.multiArrayValue![0].doubleValue)
}

let (coreModel, compiledModelUrl) = compileCoreML(path: "/Users/jacopo/TestCoreMLTools/model/coreml_model_double_train.mlmodel")

// print(coreModel.modelDescription)

let record: [Float] = [-0.38987719,  0.06312032, -0.48184431, -0.27218959, -0.25178915, -0.38486309,
                       0.63271075,  1.27556958, -0.51912059, -0.5780854,  -1.51173314,  0.33334089, 0.62918333] // Expected 18.9

let record2: [Float] = [-0.31519425, -0.48782211, -0.44154073, -0.27218959, -0.13064423, -0.97135058,
                        0.62554448,  0.28698074, -0.63514919, -0.60246703,  1.19693287, -0.57894038, 0.54597083] // Expected 13.9

let record17: [Float] = [-0.4044373,  -0.48782211, -1.14759972, -0.27218959, -0.55465147,  0.19583194,
                         0.20273463, -0.34317238, -0.86720641, -0.82799709, -0.29750355,  0.40825989, -0.62605612] // Expected 22.0

let prediction = inferenceCoreML(model: coreModel, x: record)
print(prediction)

let prediction2 = inferenceCoreML(model: coreModel, x: record2)
print(prediction2)

let prediction17 = inferenceCoreML(model: coreModel, x: record17)
print(prediction17)


