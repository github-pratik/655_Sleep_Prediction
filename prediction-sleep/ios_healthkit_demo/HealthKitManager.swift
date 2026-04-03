import Foundation
import HealthKit

struct NightSleepSummary {
    let start: Date
    let end: Date
    let sourceSummary: String
    let features: [String: Double]
}

struct QuantityStats {
    let mean: Double
    let min: Double
    let max: Double
    let median: Double
    let std: Double
    let sampleCount: Int
    let sourceNames: [String]
    let watchSampleRatio: Double
}

final class HealthKitManager: ObservableObject {
    private let healthStore = HKHealthStore()

    @Published var authorizationGranted = false
    @Published var statusMessage = "Health access not requested"
    @Published var lastDataOrigin = "Unknown"

    private let metricDescriptors: [(prefix: String, id: HKQuantityTypeIdentifier, unit: HKUnit)] = [
        ("hr", .heartRate, HKUnit.count().unitDivided(by: .minute())),
        ("hrv", .heartRateVariabilitySDNN, HKUnit.secondUnit(with: .milli)),
        ("resp", .respiratoryRate, HKUnit.count().unitDivided(by: .minute())),
        ("spo2", .oxygenSaturation, HKUnit.percent())
    ]

    func requestAuthorization(completion: @escaping (Result<Void, Error>) -> Void) {
        guard HKHealthStore.isHealthDataAvailable() else {
            statusMessage = "HealthKit not available on this device."
            completion(.failure(NSError(domain: "HealthKit", code: 1)))
            return
        }

        guard
            let sleepType = HKObjectType.categoryType(forIdentifier: .sleepAnalysis),
            let hrType = HKObjectType.quantityType(forIdentifier: .heartRate),
            let hrvType = HKObjectType.quantityType(forIdentifier: .heartRateVariabilitySDNN),
            let respType = HKObjectType.quantityType(forIdentifier: .respiratoryRate),
            let spo2Type = HKObjectType.quantityType(forIdentifier: .oxygenSaturation)
        else {
            statusMessage = "Unable to initialize HealthKit data types."
            completion(.failure(NSError(domain: "HealthKit", code: 2)))
            return
        }

        let readTypes: Set<HKObjectType> = [sleepType, hrType, hrvType, respType, spo2Type]

        healthStore.requestAuthorization(toShare: [], read: readTypes) { [weak self] success, error in
            DispatchQueue.main.async {
                self?.authorizationGranted = success
                self?.statusMessage = success ? "Health access granted" : "Health access denied"
                if let error {
                    completion(.failure(error))
                } else if success {
                    completion(.success(()))
                } else {
                    completion(.failure(NSError(domain: "HealthKit", code: 3)))
                }
            }
        }
    }

    func fetchLatestNightFeatures(completion: @escaping (Result<[String: Double], Error>) -> Void) {
        fetchLatestSleepSummary { [weak self] result in
            guard let self else { return }

            switch result {
            case .failure(let error):
                DispatchQueue.main.async {
                    self.statusMessage = "Sleep fetch failed: \(error.localizedDescription)"
                    completion(.failure(error))
                }
            case .success(let summary):
                var features = summary.features
                let group = DispatchGroup()
                let lock = NSLock()
                var metricOrigins: [String] = []
                var metricsUsingWatch = 0

                for metric in self.metricDescriptors {
                    group.enter()
                    self.fetchQuantityStats(
                        identifier: metric.id,
                        unit: metric.unit,
                        start: summary.start,
                        end: summary.end
                    ) { stats in
                        if let stats {
                            lock.lock()
                            features["\(metric.prefix)_mean"] = stats.mean
                            features["\(metric.prefix)_min"] = stats.min
                            features["\(metric.prefix)_max"] = stats.max
                            features["\(metric.prefix)_median"] = stats.median
                            features["\(metric.prefix)_std"] = stats.std
                            if stats.watchSampleRatio > 0 {
                                metricsUsingWatch += 1
                            }
                            let sourceLabel = stats.sourceNames.isEmpty ? "unknown source" : stats.sourceNames.joined(separator: ", ")
                            let pct = Int((stats.watchSampleRatio * 100.0).rounded())
                            metricOrigins.append("\(metric.prefix):\(sourceLabel) watch=\(pct)% n=\(stats.sampleCount)")
                            lock.unlock()
                        }
                        group.leave()
                    }
                }

                group.notify(queue: .main) {
                    self.lastDataOrigin = [
                        summary.sourceSummary,
                        "watch-priority metrics \(metricsUsingWatch)/\(self.metricDescriptors.count)"
                    ].joined(separator: " | ")
                    self.statusMessage = metricOrigins.isEmpty
                        ? "Latest night features fetched"
                        : "Latest night features fetched (\(metricOrigins.joined(separator: " | ")))"
                    completion(.success(features))
                }
            }
        }
    }

    private func isWatchSource(_ source: HKSource) -> Bool {
        let name = source.name.lowercased()
        let bundle = source.bundleIdentifier.lowercased()
        return name.contains("watch") || bundle.contains("watch")
    }

    private func isUserEntered(_ metadata: [String: Any]?) -> Bool {
        guard let metadata else { return false }
        return (metadata[HKMetadataKeyWasUserEntered] as? Bool) ?? false
    }

    private func fetchLatestSleepSummary(completion: @escaping (Result<NightSleepSummary, Error>) -> Void) {
        guard let sleepType = HKObjectType.categoryType(forIdentifier: .sleepAnalysis) else {
            completion(.failure(NSError(domain: "HealthKit", code: 4)))
            return
        }

        let now = Date()
        let start = Calendar.current.date(byAdding: .day, value: -3, to: now) ?? now.addingTimeInterval(-3 * 24 * 3600)
        let predicate = HKQuery.predicateForSamples(withStart: start, end: now, options: [])
        let sort = [NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: true)]

        let query = HKSampleQuery(
            sampleType: sleepType,
            predicate: predicate,
            limit: HKObjectQueryNoLimit,
            sortDescriptors: sort
        ) { _, samples, error in
            if let error {
                completion(.failure(error))
                return
            }

            guard let sleepSamples = samples as? [HKCategorySample], !sleepSamples.isEmpty else {
                completion(.failure(NSError(domain: "HealthKit", code: 5)))
                return
            }

            let nonUserSamples = sleepSamples.filter { !self.isUserEntered($0.metadata) }
            let usableSleep = nonUserSamples.isEmpty ? sleepSamples : nonUserSamples

            // Group samples by "night date" using a -6 hour shift.
            let calendar = Calendar.current
            var grouped: [Date: [HKCategorySample]] = [:]
            for sample in usableSleep {
                let shifted = calendar.date(byAdding: .hour, value: -6, to: sample.startDate) ?? sample.startDate
                let nightKey = calendar.startOfDay(for: shifted)
                grouped[nightKey, default: []].append(sample)
            }

            guard let latestNight = grouped.keys.max(), let nightSamples = grouped[latestNight], !nightSamples.isEmpty else {
                completion(.failure(NSError(domain: "HealthKit", code: 6)))
                return
            }

            let watchNight = nightSamples.filter { self.isWatchSource($0.sourceRevision.source) }
            let selectedNightSamples = watchNight.isEmpty ? nightSamples : watchNight

            let sleepStart = selectedNightSamples.map(\.startDate).min() ?? latestNight
            let sleepEnd = selectedNightSamples.map(\.endDate).max() ?? latestNight

            var inBedMinutes = 0.0
            var remMinutes = 0.0
            var deepMinutes = 0.0
            var coreMinutes = 0.0
            var asleepMinutes = 0.0

            for sample in selectedNightSamples {
                let value = sample.value
                let minutes = sample.endDate.timeIntervalSince(sample.startDate) / 60.0
                if minutes <= 0 { continue }

                if value == HKCategoryValueSleepAnalysis.inBed.rawValue {
                    inBedMinutes += minutes
                }

                // Sleep stage values from HealthKit:
                // 1: asleepUnspecified, 3: asleepCore, 4: asleepDeep, 5: asleepREM
                if value == 1 {
                    asleepMinutes += minutes
                } else if value == 3 {
                    coreMinutes += minutes
                    asleepMinutes += minutes
                } else if value == 4 {
                    deepMinutes += minutes
                    asleepMinutes += minutes
                } else if value == 5 {
                    remMinutes += minutes
                    asleepMinutes += minutes
                }
            }

            let totalSleepMinutes = asleepMinutes
            let sleepEfficiency = inBedMinutes > 0 ? (asleepMinutes / inBedMinutes) : 0
            let remPct = totalSleepMinutes > 0 ? remMinutes / totalSleepMinutes : 0
            let deepPct = totalSleepMinutes > 0 ? deepMinutes / totalSleepMinutes : 0
            let corePct = totalSleepMinutes > 0 ? coreMinutes / totalSleepMinutes : 0

            let features: [String: Double] = [
                "in_bed_minutes": inBedMinutes,
                "rem_minutes": remMinutes,
                "deep_minutes": deepMinutes,
                "core_minutes": coreMinutes,
                "asleep_minutes": asleepMinutes,
                "total_sleep_minutes": totalSleepMinutes,
                "sleep_efficiency": sleepEfficiency,
                "rem_pct": remPct,
                "deep_pct": deepPct,
                "core_pct": corePct
            ]

            let sleepSources = Array(Set(selectedNightSamples.map { $0.sourceRevision.source.name })).sorted()
            let sourceSummary = watchNight.isEmpty
                ? "sleep source: Health app merged"
                : "sleep source: Apple Watch preferred (\(sleepSources.joined(separator: ", ")))"

            completion(
                .success(
                    NightSleepSummary(
                        start: sleepStart,
                        end: sleepEnd,
                        sourceSummary: sourceSummary,
                        features: features
                    )
                )
            )
        }

        healthStore.execute(query)
    }

    private func fetchQuantityStats(
        identifier: HKQuantityTypeIdentifier,
        unit: HKUnit,
        start: Date,
        end: Date,
        completion: @escaping (QuantityStats?) -> Void
    ) {
        guard let quantityType = HKObjectType.quantityType(forIdentifier: identifier) else {
            completion(nil)
            return
        }

        let predicate = HKQuery.predicateForSamples(withStart: start, end: end, options: [.strictStartDate, .strictEndDate])
        let query = HKSampleQuery(
            sampleType: quantityType,
            predicate: predicate,
            limit: HKObjectQueryNoLimit,
            sortDescriptors: nil
        ) { _, samples, _ in
            guard let rawSamples = samples as? [HKQuantitySample], !rawSamples.isEmpty else {
                completion(nil)
                return
            }

            let nonUserSamples = rawSamples.filter { !self.isUserEntered($0.metadata) }
            let usableSamples = nonUserSamples.isEmpty ? rawSamples : nonUserSamples
            let watchSamples = usableSamples.filter { self.isWatchSource($0.sourceRevision.source) }
            let selectedSamples = watchSamples.isEmpty ? usableSamples : watchSamples
            let values = selectedSamples
                .map({ $0.quantity.doubleValue(for: unit) })
                .filter({ $0.isFinite })
            guard !values.isEmpty else {
                completion(nil)
                return
            }

            let sorted = values.sorted()
            let count = Double(values.count)
            let mean = values.reduce(0, +) / count
            let minVal = sorted.first ?? mean
            let maxVal = sorted.last ?? mean
            let median: Double = {
                if sorted.count % 2 == 1 {
                    return sorted[sorted.count / 2]
                }
                let i = sorted.count / 2
                return (sorted[i - 1] + sorted[i]) / 2.0
            }()
            let variance = values.reduce(0.0) { $0 + pow($1 - mean, 2) } / count
            let std = sqrt(max(variance, 0))
            let sourceNames = Array(Set(selectedSamples.map { $0.sourceRevision.source.name })).sorted()
            let watchRatio = usableSamples.isEmpty ? 0.0 : Double(watchSamples.count) / Double(usableSamples.count)

            completion(
                QuantityStats(
                    mean: mean,
                    min: minVal,
                    max: maxVal,
                    median: median,
                    std: std,
                    sampleCount: selectedSamples.count,
                    sourceNames: sourceNames,
                    watchSampleRatio: watchRatio
                )
            )
        }

        healthStore.execute(query)
    }
}
