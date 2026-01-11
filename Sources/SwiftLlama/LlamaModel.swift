import Foundation
import llama

class LlamaModel {
    private let model: Model
    private let vocab: Vocab
    private let configuration: Configuration
    private let context: OpaquePointer
    private let sampler: UnsafeMutablePointer<llama_sampler>
    private var batch: Batch
    private var tokens: [Token]
    private var generatedTokenAccount: Int32 = 0
    private var ended = false

    var shouldContinue: Bool {
        generatedTokenAccount < configuration.maxTokenCount && !ended
    }

    init(path: String, configuration: Configuration = .init()) throws {
        self.configuration = configuration
        llama_backend_init()
        llama_numa_init(GGML_NUMA_STRATEGY_DISABLED)

        var model_params = llama_model_default_params()
        #if targetEnvironment(simulator)
        model_params.n_gpu_layers = 0
        #endif

        guard let model = llama_model_load_from_file(path, model_params) else {
            throw SwiftLlamaError.others("Cannot load model at path \(path)")
        }
        self.model = model

        guard let vocab = llama_model_get_vocab(model) else {
            throw SwiftLlamaError.others("Cannot read model vocabulary")
        }
        self.vocab = vocab

        guard let context = llama_init_from_model(model, configuration.contextParameters) else {
            throw SwiftLlamaError.others("Cannot load model context")
        }
        self.context = context

        self.tokens = []
        self.batch = llama_batch_init(Int32(configuration.batchSize * Configuration.historySize * 2), 0, 1)

        self.sampler = llama_sampler_chain_init(llama_sampler_chain_default_params())
//        llama_sampler_chain_add(sampler, llama_sampler_init_temp(configuration.temperature))
//        llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234))
        
        // Repetition penalty
        llama_sampler_chain_add(
            sampler,
            llama_sampler_init_penalties(
                Int32(64),                        // penalty_last_n
                configuration.repeatPenalty,     // penalty_repeat (1.05)
                0.0,                              // penalty_freq
                0.0                               // penalty_present
            )
        )

        // Top-K
        llama_sampler_chain_add(
            sampler,
            llama_sampler_init_top_k(Int32(configuration.topK))
        )

        // Top-P
        llama_sampler_chain_add(
            sampler,
            llama_sampler_init_top_p(configuration.topP, 1)
        )

        // Temperature
        llama_sampler_chain_add(
            sampler,
            llama_sampler_init_temp(configuration.temperature)
        )

        // RNG / distribution
        llama_sampler_chain_add(
            sampler,
            llama_sampler_init_dist(UInt32(configuration.seed))
        )

        try checkContextLength(context: context, model: model)
    }

    private func checkContextLength(context: Context, model: Model) throws {
        let n_ctx = llama_n_ctx(context)
        let n_ctx_train = llama_model_n_ctx_train(model)
        if n_ctx > n_ctx_train {
            throw SwiftLlamaError.others("Model was trained on \(n_ctx_train) context but tokens \(n_ctx) specified")
        }
    }

    func start(for prompt: Prompt) throws {
        ended = false
        tokens = tokenize(text: prompt.prompt, addBos: true)

        batch.clear()
        tokens.enumerated().forEach { index, token in
            batch.add(token: token, position: Int32(index), seqIDs: [0], logit: false)
        }
        batch.logits[Int(batch.n_tokens) - 1] = 1 // true

        if llama_decode(context, batch) != 0 {
            throw SwiftLlamaError.decodeError
        }
        generatedTokenAccount = batch.n_tokens
    }

    func `continue`() throws -> String {
        let newToken = llama_sampler_sample(sampler, context, batch.n_tokens - 1)

        if llama_vocab_is_eog(vocab, newToken) ||
           generatedTokenAccount >= configuration.maxTokenCount {
            ended = true
            return ""
        }

        let piece = tokenToString(token: newToken)

        batch.clear()
        batch.add(token: newToken, position: generatedTokenAccount, seqIDs: [0], logit: true)
        generatedTokenAccount += 1

        if llama_decode(context, batch) != 0 {
            throw SwiftLlamaError.decodeError
        }
        return piece
    }

    // MARK: - Helpers

    /// Convert a sampled token to a Swift String (valid UTF-8, no interleaved \0 bytes).
    private func tokenToString(token: llama_token) -> String {
        var cap: Int32 = 32
        var buf = [CChar](repeating: 0, count: Int(cap))

        // First attempt
        var written: Int32 = buf.withUnsafeMutableBufferPointer { p in
            guard let base = p.baseAddress else { return 0 }
            return Int32(llama_token_to_piece(vocab, token, base, cap, 0, false))
        }

        // If negative, allocate required size and retry
        if written < 0 {
            cap = -written
            buf = [CChar](repeating: 0, count: Int(cap))
            written = buf.withUnsafeMutableBufferPointer { p -> Int32 in
                guard let base = p.baseAddress else { return 0 }
                return Int32(llama_token_to_piece(vocab, token, base, cap, 0, false))
            }
        }

        let count = Int(max(0, written))
        if count == 0 { return "" }

        // Decode exact byte count (no trailing NUL included)
        let bytes: [UInt8] = buf.prefix(count).map { UInt8(bitPattern: $0) }
        return String(decoding: bytes, as: UTF8.self)
    }

    private func tokenize(text: String, addBos: Bool) -> [Token] {
        let utf8Count = text.utf8.count
        let n_tokens = utf8Count + (addBos ? 1 : 0) + 1

        return Array(unsafeUninitializedCapacity: n_tokens) { buffer, initializedCount in
            initializedCount = Int(
                llama_tokenize(vocab, text, Int32(utf8Count), buffer.baseAddress, Int32(n_tokens), addBos, false)
            )
        }
    }

    func clear() {
        tokens.removeAll()
        if let mem = llama_get_memory(context) {
            llama_memory_clear(mem, true)
        }
    }

    deinit {
        llama_batch_free(batch)
        llama_free(context)
        llama_model_free(model)
        llama_backend_free()
    }
}
