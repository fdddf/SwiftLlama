import Foundation

public struct Prompt {
    // Simplified Prompt struct that only stores the raw components
    // The actual formatting will be handled by the model's native chat template
    public let systemPrompt: String
    public let userMessage: String
    public let history: [Chat]

    public init(systemPrompt: String = "",
                userMessage: String,
                history: [Chat] = []) {
        self.systemPrompt = systemPrompt
        self.userMessage = userMessage
        self.history = history
    }
    
    // Keep the old computed property for backward compatibility
    // but this will no longer be used for formatting since we use model templates
    var prompt: String {
        userMessage  // Return just the raw user message
    }
}
