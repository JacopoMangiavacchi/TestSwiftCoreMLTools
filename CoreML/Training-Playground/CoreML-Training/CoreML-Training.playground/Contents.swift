import Foundation
import CoreML
import SwiftCoreMLTools

struct BostonHousing {
    let trainPercentage:Float = 0.8
    let numRecords: Int
    let numColumns: Int
    let numTrainRecords: Int
    let numTestRecords: Int
    let xTrain: [[Float]]
    let yTrain: [[Float]]
    let xTest: [[Float]]
    let yTest: [[Float]]

    init(path: String) {
        let data = try! String(contentsOfFile: path, encoding: String.Encoding.utf8)

        // Convert Space Separated CSV with no Header
        let dataRecords: [[Float]] = data.split(separator: "\n").map{ String($0).split(separator: " ").compactMap{ Float(String($0)) } }

        let numRecords = dataRecords.count
        let numColumns = dataRecords[0].count

        let dataFeatures = dataRecords.map{ Array($0[0..<numColumns-1]) }
        let dataLabels = dataRecords.map{ Array($0[(numColumns-1)...]) }

        self.numRecords = numRecords
        self.numColumns = numColumns
        self.numTrainRecords = Int(ceil(Float(numRecords) * trainPercentage))
        self.numTestRecords = numRecords - numTrainRecords

        self.yTrain = Array(dataLabels[0..<numTrainRecords])
        self.yTest = Array(dataLabels[numTrainRecords...])
        var xTrainNormalized = Array(dataFeatures[0..<numTrainRecords])
        var xTestNormalized = Array(dataFeatures[numTrainRecords...])

        // Normalize
        
        var mean = Array(repeating: Float(0), count: numColumns - 1)
        for r in xTrainNormalized {
            for c in 0..<mean.count {
                mean[c] = mean[c] + r[c]
            }
        }
        for c in 0..<mean.count {
            mean[c] = mean[c] / Float(numTrainRecords)
        }

        var std = Array(repeating: Float(0), count: numColumns - 1)
        for r in xTrainNormalized {
            for c in 0..<mean.count {
                std[c] = std[c] + pow(r[c] - mean[c], 2.0)
            }
        }
        for c in 0..<mean.count {
            std[c] = std[c] / Float(numTrainRecords - 1)
        }

        for r in 0..<xTrainNormalized.count {
            for c in 0..<(numColumns - 1) {
                xTrainNormalized[r][c] = (xTrainNormalized[r][c] - mean[c]) / std[c]
            }
        }

        for r in 0..<xTestNormalized.count {
            for c in 0..<(numColumns - 1) {
                xTestNormalized[r][c] = (xTestNormalized[r][c] - mean[c]) / std[c]
            }
        }
        
//        print(mean)
//        print(std)

        self.xTrain = xTrainNormalized
        self.xTest = xTestNormalized
    }
    
    func saveModel(path: String) {
        let coremlModel = Model(version: 4,
                                shortDescription: "Regression",
                                author: "Jacopo Mangiavacchi",
                                license: "MIT",
                                userDefined: ["SwiftCoremltoolsVersion" : "0.0.3"]) {
            Input(name: "input", shape: [13], featureType: .Double)
            Output(name: "output", shape: [1], featureType: .Double)
            NeuralNetwork {
                InnerProduct(name: "dense1",
                             input: ["input"],
                             output: ["outDense1"],
                             weights: nil,
                             bias: nil,
                             inputChannels: 13,
                             outputChannels: 64,
                             updatable: false)
                ReLu(name: "Relu1",
                     input: ["outDense1"],
                     output: ["outRelu1"])
                InnerProduct(name: "dense2",
                             input: ["outRelu1"],
                             output: ["outDense2"],
                             weights: nil,
                             bias: nil,
                             inputChannels: 64,
                             outputChannels: 32,
                             updatable: false)
                ReLu(name: "Relu2",
                     input: ["outDense2"],
                     output: ["outRelu2"])
                InnerProduct(name: "dense3",
                             input: ["outRelu2"],
                             output: ["output"],
                             weights: nil,
                             bias: nil,
                             inputChannels: 32,
                             outputChannels: 1,
                             updatable: false)
            }
        }

        let coreMLData = coremlModel.coreMLData
        try! coreMLData!.write(to: URL(fileURLWithPath: path))
    }
}



let folderPath = "/Users/jacopo/TestSwiftCoreMLTools/"

let boston = BostonHousing(path: folderPath + "data/housing.csv")

boston.saveModel(path: folderPath + "model/coreml_no_trained.mlmodel")





print(boston.xTrain[0], boston.yTrain[0])
print(boston.xTest[0], boston.yTest[0])
print(boston.xTest[1], boston.yTest[1])
print(boston.xTest[17], boston.yTest[17])

