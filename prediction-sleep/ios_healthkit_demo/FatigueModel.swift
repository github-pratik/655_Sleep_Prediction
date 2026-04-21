import Foundation

struct LinearContract: Codable {
    let contractType: String
    let featureOrder: [String]
    let imputerMedian: [String: Double]
    let scalerMean: [String: Double]
    let scalerScale: [String: Double]
    let coef: [String: Double]
    let intercept: Double
    let classes: [Int]
    let decisionThreshold: Double?

    enum CodingKeys: String, CodingKey {
        case contractType = "contract_type"
        case featureOrder = "feature_order"
        case imputerMedian = "imputer_median"
        case scalerMean = "scaler_mean"
        case scalerScale = "scaler_scale"
        case coef
        case intercept
        case classes
        case decisionThreshold = "decision_threshold"
    }
}

struct FatiguePrediction {
    let label: Int
    let probability0: Double
    let probability1: Double
    let logit: Double
}

enum FatigueModelError: Error, LocalizedError {
    case resourceNotFound(String)
    case decodeFailed(String)

    var errorDescription: String? {
        switch self {
        case .resourceNotFound(let name):
            return "Could not find \(name).json in the app bundle (check Copy Bundle Resources)."
        case .decodeFailed(let detail):
            return "Invalid model contract: \(detail)"
        }
    }
}

final class FatigueModel {
    let contract: LinearContract

    init(contract: LinearContract) {
        self.contract = contract
    }

    init(resourceName: String = "mobile_linear_contract", bundle: Bundle = .main) throws {
        let jsonName = "\(resourceName).json"
        var url = bundle.url(forResource: resourceName, withExtension: "json")
        if url == nil, let root = bundle.resourceURL {
            let direct = root.appendingPathComponent(jsonName)
            if FileManager.default.fileExists(atPath: direct.path) {
                url = direct
            }
        }
        guard let url else {
            throw FatigueModelError.resourceNotFound(resourceName)
        }
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw FatigueModelError.resourceNotFound(resourceName)
        }
        do {
            contract = try JSONDecoder().decode(LinearContract.self, from: data)
        } catch {
            throw FatigueModelError.decodeFailed(error.localizedDescription)
        }
    }

    func completeFeatures(_ raw: [String: Double]) -> [String: Double] {
        var filled: [String: Double] = [:]
        for key in contract.featureOrder {
            if let value = raw[key], value.isFinite {
                filled[key] = value
            } else {
                filled[key] = contract.imputerMedian[key] ?? 0
            }
        }
        return filled
    }

    func predict(features rawFeatures: [String: Double]) -> FatiguePrediction {
        let features = completeFeatures(rawFeatures)
        var logit = contract.intercept

        for feature in contract.featureOrder {
            let value = features[feature] ?? 0
            let mean = contract.scalerMean[feature] ?? 0
            let scale = max(contract.scalerScale[feature] ?? 1, 1e-12)
            let weight = contract.coef[feature] ?? 0

            let normalized = (value - mean) / scale
            logit += normalized * weight
        }

        let p1 = 1 / (1 + exp(-logit))
        let p0 = 1 - p1
        let threshold = contract.decisionThreshold ?? 0.5
        let positiveClass = contract.classes.count > 1 ? contract.classes[1] : 1
        let negativeClass = contract.classes.first ?? 0
        let label = p1 >= threshold ? positiveClass : negativeClass

        return FatiguePrediction(label: label, probability0: p0, probability1: p1, logit: logit)
    }
}
