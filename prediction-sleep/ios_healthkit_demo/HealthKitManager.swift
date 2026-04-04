import Foundation
import HealthKit

// MARK: - Data Models

struct NightSleepSummary {
    let nightDate: Date
    let start: Date
    let end: Date
    let sourceSummary: String
    let features: [String: Double]
    let dataAvailability: DataAvailability
}

struct DataAvailability {
    let hasSleepData: Bool
    let hasHR: Bool
    let hasHRV: Bool
    let hasResp: Bool
    let hasSpO2: Bool
    let hasSteps: Bool
    let hasActiveEnergy: Bool
    let hasWorkout: Bool
    let earliestHealthDataDate: Date?
    let nightsWithData: Int

    /// How many days of historical Health data are available
    var availableHistoryDays: Int {
        guard let earliest = earliestHealthDataDate else { return 0 }
        let days = Calendar.current.dateComponents([.day], from: earliest, to: Date()).day ?? 0
        return max(days, 0)
    }

    /// Readiness state for prediction
    var readinessState: ReadinessState {
        if !hasSleepData { return .noData }
        if availableHistoryDays < 7 { return .collectingBaseline }
        if availableHistoryDays < 30 { return .partialPersonalization }
        return .fullPersonalization
    }
}

enum ReadinessState {
    case noData
    case collectingBaseline        // < 7 days
    case partialPersonalization    // 7-30 days
    case fullPersonalization       // 30+ days

    var userMessage: String {
        switch self {
        case .noData:
            return "No sleep data found. Wear your Apple Watch to sleep tonight and check back tomorrow."
        case .collectingBaseline:
            return "Collecting your baseline data. Personalized predictions available in \(daysRemaining) days."
        case .partialPersonalization:
            return "Predictions improving as we learn your patterns. Accuracy increases over the next \(daysRemaining) days."
        case .fullPersonalization:
            return "Your predictions are fully personalized."
        }
    }

    var daysRemaining: Int {
        switch self {
        case .collectingBaseline: return max(1, 7 - readinessDays)
        case .partialPersonalization: return max(1, 30 - readinessDays)
        default: return 0
        }
    }

    private var readinessDays: Int {
        // Placeholder — actual value set from availability check
        return 0
    }
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

// MARK: - HealthKit Manager

final class HealthKitManager: ObservableObject {
    private let healthStore = HKHealthStore()

    @Published var authorizationGranted = false
    @Published var statusMessage = "Health access not requested"
    @Published var lastDataOrigin = "Unknown"
    @Published var dataAvailability = DataAvailability(
        hasSleepData: false, hasHR: false, hasHRV: false,
        hasResp: false, hasSpO2: false, hasSteps: false,
        hasActiveEnergy: false, hasWorkout: false,
        earliestHealthDataDate: nil, nightsWithData: 0
    )

    private let metricDescriptors: [(prefix: String, id: HKQuantityTypeIdentifier, unit: HKUnit)] = [
        ("hr", .heartRate, HKUnit.count().unitDivided(by: .minute())),
        ("hrv", .heartRateVariabilitySDNN, HKUnit.secondUnit(with: .milli)),
        ("resp", .respiratoryRate, HKUnit.count().unitDivided(by: .minute())),
        ("spo2", .oxygenSaturation, HKUnit.percent())
    ]

    private let activityDescriptors: [(prefix: String, id: HKQuantityTypeIdentifier, unit: HKUnit, aggregate: Bool)] = [
        ("steps", .stepCount, HKUnit.count(), true),
        ("active_energy", .activeEnergyBurned, HKUnit.kilocalorie(), true),
        ("avg_physical_effort", .appleExerciseTime, HKUnit.minute(), false)
    ]

    func requestAuthorization(completion: @escaping (Result<Void, Error>) -> Void) {
        guard HKHealthStore.isHealthDataAvailable() else {
            statusMessage = "HealthKit not available on this device."
            completion(.failure(NSError(domain: "HealthKit", code: 1)))
            return
        }

        let typesToRead: Set<HKObjectType> = [
            HKObjectType.categoryType(forIdentifier: .sleepAnalysis)!,
            HKObjectType.quantityType(forIdentifier: .heartRate)!,
            HKObjectType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!,
            HKObjectType.quantityType(forIdentifier: .respiratoryRate)!,
            HKObjectType.quantityType(forIdentifier: .oxygenSaturation)!,
            HKObjectType.quantityType(forIdentifier: .stepCount)!,
            HKObjectType.quantityType(forIdentifier: .activeEnergyBurned)!,
            HKObjectType.quantityType(forIdentifier: .appleExerciseTime)!
        ]

        healthStore.requestAuthorization(toShare: [], read: typesToRead) { [weak self] success, error in
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

    /// Check how much historical data is available for cold-start assessment
    func checkDataAvailability(completion: @escaping (DataAvailability) -> Void) {
        var availability = DataAvailability(
            hasSleepData: false, hasHR: false, hasHRV: false,
            hasResp: false, hasSpO2: false, hasSteps: false,
            hasActiveEnergy: false, hasWorkout: false,
            earliestHealthDataDate: nil, nightsWithData: 0
        )

        let group = DispatchGroup()
        let lock = NSLock()
        var earliestDates: [Date] = []

        // Check sleep data
        group.enter()
        checkTypeExists(category: .sleepAnalysis) { exists, earliest in
            lock.lock()
            availability.hasSleepData = exists
            if let earliest { earliestDates.append(earliest) }
            lock.unlock()
            group.leave()
        }

        // Check quantity types
        let allTypes: [(inout Bool, inout Date?, HKQuantityTypeIdentifier)] = [
            (&availability.hasHR, &availability.earliestHealthDataDate, .heartRate),
            (&availability.hasHRV, &availability.earliestHealthDataDate, .heartRateVariabilitySDNN),
            (&availability.hasResp, &availability.earliestHealthDataDate, .respiratoryRate),
            (&availability.hasSpO2, &availability.earliestHealthDataDate, .oxygenSaturation),
            (&availability.hasSteps, &availability.earliestHealthDataDate, .stepCount),
            (&availability.hasActiveEnergy, &availability.earliestHealthDataDate, .activeEnergyBurned),
            (&availability.hasWorkout, &availability.earliestHealthDataDate, .appleExerciseTime)
        ]

        for tuple in allTypes {
            group.enter()
            checkQuantityExists(type: tuple.2) { exists, earliest in
                lock.lock()
                tuple.0 = exists
                if let earliest { earliestDates.append(earliest) }
                lock.unlock()
                group.leave()
            }
        }

        group.notify(queue: .main) {
            lock.lock()
            availability.earliestHealthDataDate = earliestDates.min()
            availability.nightsWithData = self.countNightsWithSleepData()
            availability.readinessState // trigger computed
            let result = availability
            lock.unlock()
            completion(result)
        }
    }

    /// Fetch multiple nights for trend analysis
    func fetchMultipleNights(count: Int = 7, completion: @escaping (Result<[NightSleepSummary], Error>) -> Void) {
        guard let sleepType = HKObjectType.categoryType(forIdentifier: .sleepAnalysis) else {
            completion(.failure(NSError(domain: "HealthKit", code: 4)))
            return
        }

        let now = Date()
        let start = Calendar.current.date(byAdding: .day, value: -(count + 3), to: now) ?? now.addingTimeInterval(-Double(count + 3) * 24 * 3600)
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

            let calendar = Calendar.current
            var grouped: [Date: [HKCategorySample]] = [:]
            for sample in usableSleep {
                let shifted = calendar.date(byAdding: .hour, value: -6, to: sample.startDate) ?? sample.startDate
                let nightKey = calendar.startOfDay(for: shifted)
                grouped[nightKey, default: []].append(sample)
            }

            let sortedNights = grouped.keys.sorted().reversed()
            let limitedNights = Array(sortedNights.prefix(count))

            var results: [NightSleepSummary] = []
            let innerGroup = DispatchGroup()
            let innerLock = NSLock()

            for nightKey in limitedNights {
                guard let nightSamples = grouped[nightKey], !nightSamples.isEmpty else { continue }
                innerGroup.enter()

                self.buildNightSummary(from: nightSamples, nightDate: nightKey, calendar: calendar) { summary in
                    if let summary {
                        innerLock.lock()
                        results.append(summary)
                        innerLock.unlock()
                    }
                    innerGroup.leave()
                }
            }

            innerGroup.notify(queue: .main) {
                results.sort { $0.nightDate > $1.nightDate }
                completion(.success(results))
            }
        }

        healthStore.execute(query)
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
                self.enrichWithActivityAndMetrics(summary: summary, completion: completion)
            }
        }
    }

    // MARK: - Private

    private func buildNightSummary(
        from nightSamples: [HKCategorySample],
        nightDate: Date,
        calendar: Calendar,
        completion: @escaping (NightSleepSummary?) -> Void
    ) {
        let watchNight = nightSamples.filter { self.isWatchSource($0.sourceRevision.source) }
        let selectedNightSamples = watchNight.isEmpty ? nightSamples : watchNight

        let sleepStart = selectedNightSamples.map(\.startDate).min() ?? nightDate
        let sleepEnd = selectedNightSamples.map(\.endDate).max() ?? nightDate

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
            NightSleepSummary(
                nightDate: nightDate,
                start: sleepStart,
                end: sleepEnd,
                sourceSummary: sourceSummary,
                features: features,
                dataAvailability: DataAvailability(
                    hasSleepData: true, hasHR: false, hasHRV: false,
                    hasResp: false, hasSpO2: false, hasSteps: false,
                    hasActiveEnergy: false, hasWorkout: false,
                    earliestHealthDataDate: nil, nightsWithData: 0
                )
            )
        )
    }

    private func enrichWithActivityAndMetrics(
        summary: NightSleepSummary,
        completion: @escaping (Result<[String: Double], Error>) -> Void
    ) {
        var features = summary.features
        let group = DispatchGroup()
        let lock = NSLock()
        var metricOrigins: [String] = []
        var metricsUsingWatch = 0

        // Fetch physio metrics
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

        // Fetch activity data for the DAY before the night
        let dayBeforeStart = Calendar.current.date(byAdding: .hour, value: -18, to: summary.start) ?? summary.start.addingTimeInterval(-18 * 3600)
        let dayBeforeEnd = summary.start

        for activity in self.activityDescriptors {
            group.enter()
            self.fetchActivityAggregate(
                identifier: activity.id,
                unit: activity.unit,
                start: dayBeforeStart,
                end: dayBeforeEnd,
                sum: activity.aggregate
            ) { value in
                if let value {
                    lock.lock()
                    features[activity.prefix] = value
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

    private func fetchActivityAggregate(
        identifier: HKQuantityTypeIdentifier,
        unit: HKUnit,
        start: Date,
        end: Date,
        sum: Bool,
        completion: @escaping (Double?) -> Void
    ) {
        guard let quantityType = HKObjectType.quantityType(forIdentifier: identifier) else {
            completion(nil)
            return
        }

        let predicate = HKQuery.predicateForSamples(withStart: start, end: end, options: [.strictStartDate, .strictEndDate])

        let query = HKStatisticsQuery(
            quantityType: quantityType,
            quantitySamplePredicate: predicate,
            options: sum ? .cumulativeSum : .discreteAverage
        ) { _, result, _ in
            guard let result, let quantity = sum ? result.sumQuantity() : result.averageQuantity() else {
                completion(nil)
                return
            }
            let value = quantity.doubleValue(for: unit)
            completion(value.isFinite ? value : nil)
        }

        healthStore.execute(query)
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

            self.buildNightSummary(from: nightSamples, nightDate: latestNight, calendar: calendar) { summary in
                if let summary {
                    completion(.success(summary))
                } else {
                    completion(.failure(NSError(domain: "HealthKit", code: 7)))
                }
            }
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

    // MARK: - Availability helpers

    private func checkTypeExists(category: HKCategoryTypeIdentifier, completion: @escaping (Bool, Date?) -> Void) {
        guard let type = HKObjectType.categoryType(forIdentifier: category) else {
            completion(false, nil)
            return
        }
        let predicate = HKQuery.predicateForSamples(withStart: Date.distantPast, end: Date.distantFuture, options: [])
        let query = HKSampleQuery(sampleType: type, predicate: predicate, limit: 1, sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: true)]) { _, samples, _ in
            let earliest = (samples as? [HKCategorySample])?.first?.startDate
            completion(samples?.isEmpty == false, earliest)
        }
        healthStore.execute(query)
    }

    private func checkQuantityExists(type: HKQuantityTypeIdentifier, completion: @escaping (Bool, Date?) -> Void) {
        guard let quantityType = HKObjectType.quantityType(forIdentifier: type) else {
            completion(false, nil)
            return
        }
        let predicate = HKQuery.predicateForSamples(withStart: Date.distantPast, end: Date.distantFuture, options: [])
        let query = HKSampleQuery(sampleType: quantityType, predicate: predicate, limit: 1, sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: true)]) { _, samples, _ in
            let earliest = (samples as? [HKQuantitySample])?.first?.startDate
            completion(samples?.isEmpty == false, earliest)
        }
        healthStore.execute(query)
    }

    private func countNightsWithSleepData() -> Int {
        // Synchronous placeholder — actual count from grouped samples
        return 0
    }
}
