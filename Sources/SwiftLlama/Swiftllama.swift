import Foundation
import llama
import Combine

public class SwiftLlama {
    private let model: LlamaModel
    private let configuration: Configuration
    private var contentStarted = false
    private var sessionSupport = false {
        didSet {
            if !sessionSupport {
                session = nil
            }
        }
    }

    private var session: Session?
    private lazy var resultSubject: CurrentValueSubject<String, Error> = {
        .init("")
    }()
    private var generatedTokenCache = ""

    var maxLengthOfStopToken: Int {
        configuration.stopTokens.map { $0.count }.max() ?? 0
    }

    public init(modelPath: String,
                 modelConfiguration: Configuration = .init()) throws {
        self.model = try LlamaModel(path: modelPath, configuration: modelConfiguration)
        self.configuration = modelConfiguration
    }

    private func prepare(sessionSupport: Bool, for prompt: Prompt) -> Prompt {
        contentStarted = false
        generatedTokenCache = ""
        self.sessionSupport = sessionSupport
        if sessionSupport {
            if session == nil {
                session = Session(lastPrompt: prompt)
            } else {
                session?.lastPrompt = prompt
            }
            return session?.sessionPrompt ?? prompt
        } else {
            return prompt
        }
    }

    private func isStopToken() -> Bool {
        configuration.stopTokens.reduce(false) { partialResult, stopToken in
            generatedTokenCache.hasSuffix(stopToken)
        }
    }

    private func response(for prompt: Prompt, output: (String) -> Void, finish: () -> Void) {
        func finaliseOutput() {
            configuration.stopTokens.forEach {
                generatedTokenCache = generatedTokenCache.replacingOccurrences(of: $0, with: "")
            }
            output(generatedTokenCache)
            finish()
            generatedTokenCache = ""
        }
        defer { model.clear() }
        do {
            try model.start(for: prompt)
            while model.shouldContinue {
                var delta = try model.continue()
                if contentStarted { // remove the prefix empty spaces
                    if needToStop(after: delta, output: output) {
                        finish()
                        break
                    }
                } else {
                    delta = delta.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !delta.isEmpty {
                        contentStarted = true
                        if needToStop(after: delta, output: output) {
                            finish()
                            break
                        }
                    }
                }
            }
            finaliseOutput()
        } catch {
            finaliseOutput()
        }
    }

    /// Handling logic of StopToken
    private func needToStop(after delta: String, output: (String) -> Void) -> Bool {
        // If no stop tokens, just stream through
        guard maxLengthOfStopToken > 0 else {
            output(delta)
            return false
        }

        generatedTokenCache += delta

        // 1) If any stop token appears, cut output before it and stop
        if let stopRange = configuration.stopTokens
            .compactMap({ generatedTokenCache.range(of: $0) })
            .min(by: { $0.lowerBound < $1.lowerBound }) // earliest occurrence
        {
            let before = String(generatedTokenCache[..<stopRange.lowerBound])
            if !before.isEmpty { output(before) }
            generatedTokenCache.removeAll(keepingCapacity: false)
            return true
        }

        // 2) Stream everything except a small tail so split stop tokens are caught next time
        let tail = max(maxLengthOfStopToken - 1, 0)
        if generatedTokenCache.count > tail {
            let cut = generatedTokenCache.index(generatedTokenCache.endIndex, offsetBy: -tail)
            let safe = String(generatedTokenCache[..<cut])
            if !safe.isEmpty { output(safe) }
            generatedTokenCache.removeFirst(safe.count)
        }

        return false
    }

    @SwiftLlamaActor
    public func start(for prompt: Prompt, sessionSupport: Bool = false) -> AsyncThrowingStream<String, Error> {
        let sessionPrompt = prepare(sessionSupport: sessionSupport, for: prompt)
        return .init { continuation in
            Task {
                response(for: sessionPrompt) { [weak self] delta in
                    continuation.yield(delta)
                    self?.session?.response(delta: delta)
                } finish: { [weak self] in
                    continuation.finish()
                    self?.session?.endResponse()
                }
            }
        }
    }

    @SwiftLlamaActor
    public func start(for prompt: Prompt, sessionSupport: Bool = false) -> AnyPublisher<String, Error> {
        let sessionPrompt = prepare(sessionSupport: sessionSupport, for: prompt)
        Task {
            response(for: sessionPrompt) { delta in
                resultSubject.send(delta)
                session?.response(delta: delta)
            } finish: {
                resultSubject.send(completion: .finished)
                session?.endResponse()
            }
        }
        return resultSubject.eraseToAnyPublisher()
    }

    @SwiftLlamaActor
    public func start(for prompt: Prompt, sessionSupport: Bool = false) async throws -> String {
        var result = ""
        for try await value in start(for: prompt) {
            result += value
        }
        return result
    }
    
    // MARK: - KV Cache Management
    
    /// Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
    public func removeTokens(from seqId: llama_seq_id = -1, startPos: llama_pos = -1, endPos: llama_pos = -1) -> Bool {
        return model.removeTokens(from: seqId, startPos: startPos, endPos: endPos)
    }
    
    /// Copy all tokens that belong to the specified sequence to another sequence
    public func copyTokens(from seqIdSrc: llama_seq_id, to seqIdDst: llama_seq_id, startPos: llama_pos = -1, endPos: llama_pos = -1) {
        model.copyTokens(from: seqIdSrc, to: seqIdDst, startPos: startPos, endPos: endPos)
    }
    
    /// Removes all tokens that do not belong to the specified sequence
    public func keepTokens(in seqId: llama_seq_id) {
        model.keepTokens(in: seqId)
    }
    
    /// Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
    public func addPositionDelta(to seqId: llama_seq_id, startPos: llama_pos = -1, endPos: llama_pos = -1, delta: llama_pos) {
        model.addPositionDelta(to: seqId, startPos: startPos, endPos: endPos, delta: delta)
    }
    
    /// Integer division of the positions by factor of `d > 1`
    public func dividePositions(in seqId: llama_seq_id, startPos: llama_pos = -1, endPos: llama_pos = -1, factor: Int32) {
        model.dividePositions(in: seqId, startPos: startPos, endPos: endPos, factor: factor)
    }
    
    /// Returns the smallest position present in the memory for the specified sequence
    public func getPositionMin(for seqId: llama_seq_id) -> llama_pos {
        return model.getPositionMin(for: seqId)
    }
    
    /// Returns the largest position present in the memory for the specified sequence
    public func getPositionMax(for seqId: llama_seq_id) -> llama_pos {
        return model.getPositionMax(for: seqId)
    }
    
    /// Check if the memory supports shifting
    public func canMemoryShift() -> Bool {
        return model.canMemoryShift()
    }
    
    @SwiftLlamaActor
    public func calculateTokenCount(
        for prompt: Prompt,
        sessionSupport: Bool = false,
        addBos: Bool = true
    ) throws -> Int {
        let preparedPrompt = prepare(
            sessionSupport: sessionSupport,
            for: prompt
        )

        return try model.calculateTokenCount(
            for: preparedPrompt,
            addBos: addBos
        )
    }
}
