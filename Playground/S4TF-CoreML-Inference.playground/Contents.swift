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

let record: [Float] = [-0.4013274553772651,-0.48782210614046656,-1.1729760489283325,-0.2721895900613162,-0.8055945265354896,0.09156749405417394,-1.828902543802867,0.6384789935042571,-0.6351491942719604,0.1472680456555187,-0.7178137893787737,0.2073805740660824,-0.7473489168521552] // Expected 22.6 !!!

let prediction = inferenceCoreML(model: coreModel, x: record)
print(prediction) // 22.6
