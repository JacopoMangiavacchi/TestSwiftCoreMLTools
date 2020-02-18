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
    
    let multiArr = try! MLMultiArray(shape: [13], dataType: .float32)
    for i in 0..<13 {
        multiArr[i] = NSNumber(value: x[i])
    }

    let inputValue = MLFeatureValue(multiArray: multiArr)
    let dataPointFeatures: [String: MLFeatureValue] = [inputName: inputValue]
    let provider = try! MLDictionaryFeatureProvider(dictionary: dataPointFeatures)
    
    let prediction = try! model.prediction(from: provider)

    return Float(prediction.featureValue(for: "output")!.multiArrayValue![0].floatValue)
}

let (coreModel, compiledModelUrl) = compileCoreML(path: "/Users/jacopo/TestSwiftCoreMLTools/model/s4tf_train_model.mlmodel")

// print(coreModel.modelDescription)

let cheat0: [Float] = [-0.30846816,   0.15054052,   -1.1039964,  -0.30756173,   0.05126577,   0.33352128, 0.024851447, -0.035726987,  -0.89119065,  -0.43651816,   -1.2319369,   0.42111015, -0.93603975]

let record0: [Float] = [10.127699, -0.56208307,   1.3125796, -0.30756173,   1.4050479,  -0.8863933,   1.2248203, -1.2581577,   2.6233904,   2.3634233,   0.9779132,  0.12650238,   1.6906141]

let record1: [Float] = [2.8739426, -0.56208307,   1.3125796, -0.30756173,    1.108089,  -2.9993627,   1.2248203, -1.3716108,   2.6233904,   2.3634233,   0.9779132, -0.23774466,   1.7431473]

let record17: [Float] = [1.541963, -0.56208307,   1.3125796, -0.30756173,   0.7150559, -0.93426037,   0.7972452, -1.0169381,   2.6233904,   2.3634233,   0.9779132,   -2.180478,  0.39479825]


print(inferenceCoreML(model: coreModel, x: cheat0))
print(inferenceCoreML(model: coreModel, x: record0))
print(inferenceCoreML(model: coreModel, x: record1))
print(inferenceCoreML(model: coreModel, x: record17))


