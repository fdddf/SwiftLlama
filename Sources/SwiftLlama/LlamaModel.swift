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
    private var pendingUTF8Bytes: [UInt8] = []

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
        self.batch = llama_batch_init(Int32(configuration.batchSize), 0, 1)

        self.sampler = llama_sampler_chain_init(llama_sampler_chain_default_params())
        
        // Add XTC sampler if configuration supports it
        if configuration.useXTCSampler {
            llama_sampler_chain_add(
                sampler,
                llama_sampler_init_xtc(
                    configuration.xtcProbability,  // p
                    configuration.xtcThreshold,    // t
                    1,                             // min_keep
                    UInt32(configuration.seed)     // seed
                )
            )
        }
        
        // Add DRY sampler if configuration supports it
        if configuration.useDRYSampler {
            // Note: llama_sampler_init_dry is not available in the current Swift interface
            // For now, we'll use a placeholder implementation
            // A complete implementation would require a more sophisticated approach to handle the string array
            print("DRY sampler is configured but not currently implemented in this version")
        }

        // Add repetition penalties
        llama_sampler_chain_add(
            sampler,
            llama_sampler_init_penalties(
                Int32(configuration.penaltyLastN),           // penalty_last_n
                configuration.repeatPenalty,                // penalty_repeat
                configuration.penaltyFreq,                  // penalty_freq
                configuration.penaltyPresent                // penalty_present
            )
        )

        // Top-K sampling
        if configuration.topK > 0 {
            llama_sampler_chain_add(
                sampler,
                llama_sampler_init_top_k(Int32(configuration.topK))
            )
        }

        // Top-P sampling
        if configuration.topP > 0 {
            llama_sampler_chain_add(
                sampler,
                llama_sampler_init_top_p(configuration.topP, 1)
            )
        }

        // Minimum P sampling if enabled
        if configuration.useMinP {
            llama_sampler_chain_add(
                sampler,
                llama_sampler_init_min_p(configuration.minP, 1)
            )
        }

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
        pendingUTF8Bytes.removeAll()
        
        let formattedText = formatPromptWithModelTemplate(prompt: prompt)
        
        // Check if the formatted text is empty
        guard !formattedText.isEmpty else {
            throw SwiftLlamaError.others("Formatted prompt is empty")
        }
        
        tokens = tokenize(text: formattedText, addBos: true)
        
        // Check if we have tokens
        guard !tokens.isEmpty else {
            throw SwiftLlamaError.others("No tokens generated from prompt")
        }
        
        batch.clear()
        tokens.enumerated().forEach { index, token in
            batch.add(token: token, position: Int32(index), seqIDs: [0], logit: false)
        }
        batch.logits[Int(batch.n_tokens) - 1] = 1 // true
        
        let decodeResult = llama_decode(context, batch)
        if decodeResult != 0 {
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
        let bytes = tokenToBytes(token: token)
        guard !bytes.isEmpty else { return "" }

        pendingUTF8Bytes.append(contentsOf: bytes)

        let maxTail = min(3, pendingUTF8Bytes.count)
        for tail in 0...maxTail {
            let count = pendingUTF8Bytes.count - tail
            guard count > 0 else { continue }

            if let decoded = String(bytes: pendingUTF8Bytes.prefix(count), encoding: .utf8) {
                pendingUTF8Bytes.removeFirst(count)
                return decoded
            }
        }

        return ""
    }

    private func tokenToBytes(token: llama_token) -> [UInt8] {
        var cap: Int32 = 32
        var buf = [CChar](repeating: 0, count: Int(cap))

        var written: Int32 = buf.withUnsafeMutableBufferPointer { p in
            guard let base = p.baseAddress else { return 0 }
            return Int32(llama_token_to_piece(vocab, token, base, cap, 0, false))
        }

        if written < 0 {
            cap = -written
            buf = [CChar](repeating: 0, count: Int(cap))
            written = buf.withUnsafeMutableBufferPointer { p -> Int32 in
                guard let base = p.baseAddress else { return 0 }
                return Int32(llama_token_to_piece(vocab, token, base, cap, 0, false))
            }
        }

        let count = Int(max(0, written))
        if count == 0 { return [] }

        return buf.prefix(count).map { UInt8(bitPattern: $0) }
    }

    private func tokenize(text: String, addBos: Bool) -> [Token] {
        // The text parameter here is already formatted by the Prompt struct
        // Do not apply any further formatting or templating.
        let processedText = text
        
        let utf8Count = processedText.utf8.count
        let n_tokens = utf8Count + (addBos ? 1 : 0) + 1

        return Array(unsafeUninitializedCapacity: n_tokens) { buffer, initializedCount in
            initializedCount = Int(
                // Enable parsing special tokens so model chat templates (e.g., Hunyuan) tokenize correctly.
                llama_tokenize(vocab, processedText, Int32(utf8Count), buffer.baseAddress, Int32(n_tokens), addBos, true)
            )
        }
    }
    
    private func formatPromptWithModelTemplate(prompt: Prompt) -> String {
        // Get the model's default chat template
        let templatePtr = llama_model_chat_template(model, nil)

        guard let templatePtr = templatePtr else {
            // If no template is available, fallback to a basic format
            if !prompt.systemPrompt.isEmpty {
                return "<|system|>\n\(prompt.systemPrompt)\n<|user|>\n\(prompt.userMessage)\n<|assistant|>\n"
            } else {
                return "<|user|>\n\(prompt.userMessage)\n<|assistant|>\n"
            }
        }

        // We must keep all C strings alive for the duration of llama_chat_apply_template.
        // Using String.cString(using:) returns temporary storage which becomes invalid.
        var allocated: [UnsafeMutablePointer<CChar>] = []
        func dupCString(_ s: String) -> UnsafePointer<CChar> {
            let p = strdup(s)
            allocated.append(p!)
            return UnsafePointer(p!)
        }
        defer {
            for p in allocated { free(p) }
        }

        // Prepare the chat messages for the template
        var messages: [llama_chat_message] = []

        // Add system message if system prompt is not empty
        if !prompt.systemPrompt.isEmpty {
            messages.append(
                llama_chat_message(
                    role: dupCString("system"),
                    content: dupCString(prompt.systemPrompt)
                )
            )
        }

        // Add conversation history
        for chat in prompt.history.suffix(configuration.historySize) {
            // Add user message
            messages.append(
                llama_chat_message(
                    role: dupCString("user"),
                    content: dupCString(chat.user)
                )
            )

            // Add assistant message if not empty
            if !chat.bot.isEmpty {
                messages.append(
                    llama_chat_message(
                        role: dupCString("assistant"),
                        content: dupCString(chat.bot)
                    )
                )
            }
        }

        // Add the current user message
        messages.append(
            llama_chat_message(
                role: dupCString("user"),
                content: dupCString(prompt.userMessage)
            )
        )

        // Start with a reasonable default size
        var buffer = [CChar](repeating: 0, count: 4096)

        func applyTemplate(into out: inout [CChar]) -> Int32 {
            let outCount = Int32(out.count)
            
            return messages.withUnsafeBufferPointer { msgs in
                return out.withUnsafeMutableBufferPointer { outBuf in
                    guard let outBase = outBuf.baseAddress else { return 0 }
                    return llama_chat_apply_template(
                        templatePtr,
                        msgs.baseAddress,
                        Int(Int32(msgs.count)),
                        true,  // add_ass: add assistant prefix for the model to complete
                        outBase,
                        outCount
                    )
                }
            }
            
        }

        var resultLength = applyTemplate(into: &buffer)

        // If llama returns <= 0, template application failed.
        guard resultLength > 0 else {
            return ""
        }

        // If the result doesn't fit, allocate exact required size (+1 for NUL) and retry.
        if resultLength >= Int32(buffer.count) {
            var largeBuffer = [CChar](repeating: 0, count: Int(resultLength) + 1)
            resultLength = applyTemplate(into: &largeBuffer)
            guard resultLength > 0 else { return "" }
            return String(cString: largeBuffer)
        }

        return String(cString: buffer)
    }
    
    private func formatChatTemplateIfNeeded(text: String) -> String {
        // This function is for formatting individual text pieces using the chat template
        // when they are part of a larger conversation
        return text
    }

    func clear() {
        tokens.removeAll()
        pendingUTF8Bytes.removeAll()
        if let mem = llama_get_memory(context) {
            llama_memory_clear(mem, true)
        }
    }
    
    // MARK: - KV Cache Management
    
    /// Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
    /// Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
    /// seq_id < 0 : match any sequence
    /// p0 < 0     : [0,  p1]
    /// p1 < 0     : [p0, inf)
    func removeTokens(from seqId: llama_seq_id = -1, startPos: llama_pos = -1, endPos: llama_pos = -1) -> Bool {
        if let mem = llama_get_memory(context) {
            return llama_memory_seq_rm(mem, seqId, startPos, endPos)
        }
        return false
    }
    
    /// Copy all tokens that belong to the specified sequence to another sequence
    /// p0 < 0 : [0,  p1]
    /// p1 < 0 : [p0, inf)
    func copyTokens(from seqIdSrc: llama_seq_id, to seqIdDst: llama_seq_id, startPos: llama_pos = -1, endPos: llama_pos = -1) {
        if let mem = llama_get_memory(context) {
            llama_memory_seq_cp(mem, seqIdSrc, seqIdDst, startPos, endPos)
        }
    }
    
    /// Removes all tokens that do not belong to the specified sequence
    func keepTokens(in seqId: llama_seq_id) {
        if let mem = llama_get_memory(context) {
            llama_memory_seq_keep(mem, seqId)
        }
    }
    
    /// Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
    /// p0 < 0 : [0,  p1]
    /// p1 < 0 : [p0, inf)
    func addPositionDelta(to seqId: llama_seq_id, startPos: llama_pos = -1, endPos: llama_pos = -1, delta: llama_pos) {
        if let mem = llama_get_memory(context) {
            llama_memory_seq_add(mem, seqId, startPos, endPos, delta)
        }
    }
    
    /// Integer division of the positions by factor of `d > 1`
    /// p0 < 0 : [0,  p1]
    /// p1 < 0 : [p0, inf)
    func dividePositions(in seqId: llama_seq_id, startPos: llama_pos = -1, endPos: llama_pos = -1, factor: Int32) {
        if let mem = llama_get_memory(context) {
            llama_memory_seq_div(mem, seqId, startPos, endPos, factor)
        }
    }
    
    /// Returns the smallest position present in the memory for the specified sequence
    /// This is typically non-zero only for SWA caches
    /// Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
    /// Return -1 if the sequence is empty
    func getPositionMin(for seqId: llama_seq_id) -> llama_pos {
        if let mem = llama_get_memory(context) {
            return llama_memory_seq_pos_min(mem, seqId)
        }
        return -1
    }
    
    /// Returns the largest position present in the memory for the specified sequence
    /// Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
    /// Return -1 if the sequence is empty
    func getPositionMax(for seqId: llama_seq_id) -> llama_pos {
        if let mem = llama_get_memory(context) {
            return llama_memory_seq_pos_max(mem, seqId)
        }
        return -1
    }
    
    /// Check if the memory supports shifting
    func canMemoryShift() -> Bool {
        if let mem = llama_get_memory(context) {
            return llama_memory_can_shift(mem)
        }
        return false
    }

    deinit {
        llama_batch_free(batch)
        llama_free(context)
        llama_model_free(model)
        llama_backend_free()
    }
}


extension LlamaModel {
    /// Calculates the exact number of tokens llama.cpp will consume for a prompt
    /// including chat template, special tokens, and optional BOS.
    func calculateTokenCount(
        for prompt: Prompt,
        addBos: Bool
    ) throws -> Int {
        let formattedText = formatPromptWithModelTemplate(prompt: prompt)

        guard !formattedText.isEmpty else {
            return 0
        }

        return tokenizeCount(text: formattedText, addBos: addBos)
    }
    
    private func tokenizeCount(text: String, addBos: Bool) -> Int {
        let utf8Count = text.utf8.count

        // llama.cpp-safe upper bound
        let maxTokens = utf8Count + 4 + (addBos ? 1 : 0)

        // Allocate a temporary buffer
        let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: maxTokens)
        defer { tokens.deallocate() }

        let count = llama_tokenize(
            vocab,
            text,
            Int32(utf8Count),
            tokens,
            Int32(maxTokens),
            addBos,
            true
        )

        return Int(count)
    }
}
