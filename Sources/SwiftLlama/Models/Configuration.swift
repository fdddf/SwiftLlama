import Foundation
import llama

public struct Configuration {
    public let historySize: Int
    public let seed: Int
    public let topK: Int
    public let topP: Float
    public let nCTX: Int
    public let temperature: Float
    public let maxTokenCount: Int
    public let batchSize: Int
    public let stopTokens: [String]
    public let repeatPenalty: Float
    public let addBos: Bool
    
    // New properties for advanced samplers
    public let penaltyLastN: Int
    public let penaltyFreq: Float
    public let penaltyPresent: Float
    public let useMinP: Bool
    public let minP: Float
    public let useXTCSampler: Bool
    public let xtcProbability: Float
    public let xtcThreshold: Float
    public let useDRYSampler: Bool
    public let dryMultiplier: Float
    public let dryBase: Float
    public let dryAllowedLength: Int
    public let dryPenaltyLastN: Int
    public let drySequenceBreakers: [String]

    public init(seed: Int = 1234,
                topK: Int = 40,
                topP: Float = 0.9,
                nCTX: Int = 2048,
                temperature: Float = 0.2,
                batchSize: Int = 2048,
                stopSequence: String? = nil,
                maxTokenCount: Int = 1024,
                repeatPenalty: Float = 1.0,
                addBos: Bool = true,
                stopTokens: [String] = [],
                historySize: Int = 5,
                
                // Advanced sampler parameters
                penaltyLastN: Int = 64,
                penaltyFreq: Float = 0.0,
                penaltyPresent: Float = 0.0,
                useMinP: Bool = false,
                minP: Float = 0.05,
                useXTCSampler: Bool = false,
                xtcProbability: Float = 0.0,
                xtcThreshold: Float = 0.1,
                useDRYSampler: Bool = false,
                dryMultiplier: Float = 0.0,
                dryBase: Float = 1.75,
                dryAllowedLength: Int = 2,
                dryPenaltyLastN: Int = -1,
                drySequenceBreakers: [String] = ["\n", ":", "\"", "*"]) {
        self.seed = seed
        self.topK = topK
        self.topP = topP
        self.nCTX = nCTX
        self.batchSize = batchSize
        self.temperature = temperature
        self.maxTokenCount = maxTokenCount
        self.repeatPenalty = repeatPenalty
        self.stopTokens = stopTokens
        self.addBos = addBos
        self.historySize = historySize
        
        // Advanced sampler parameters
        self.penaltyLastN = penaltyLastN
        self.penaltyFreq = penaltyFreq
        self.penaltyPresent = penaltyPresent
        self.useMinP = useMinP
        self.minP = minP
        self.useXTCSampler = useXTCSampler
        self.xtcProbability = xtcProbability
        self.xtcThreshold = xtcThreshold
        self.useDRYSampler = useDRYSampler
        self.dryMultiplier = dryMultiplier
        self.dryBase = dryBase
        self.dryAllowedLength = dryAllowedLength
        self.dryPenaltyLastN = dryPenaltyLastN
        self.drySequenceBreakers = drySequenceBreakers
    }
}

extension Configuration {
    var contextParameters: ContextParameters {
        var params = llama_context_default_params()
        let processorCount = max(1, min(16, ProcessInfo.processInfo.processorCount - 2))
        params.n_ctx = max(8, UInt32(self.nCTX)) // minimum context size is 8
        params.n_batch = UInt32(min(self.batchSize, self.nCTX))
        params.n_threads = Int32(processorCount)
        params.n_threads_batch = Int32(processorCount)
        return params
    }
}
