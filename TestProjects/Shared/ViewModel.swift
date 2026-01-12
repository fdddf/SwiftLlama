import Foundation
import SwiftLlama
import SwiftUI
import Combine

@Observable
class ViewModel {
    let swiftLlama: SwiftLlama
    var result = ""
    var usingStream = true
    private var cancallable: Set<AnyCancellable> = []
    
    struct LLamaModel {
        let modelName: String
        let stopTokens: [String]
    }
    
    static let llama2model: LLamaModel = .init(
        modelName: "llama-2-7b.Q4_K_M",
        stopTokens: StopToken.llama
    )
    
    static let hyMTModel: LLamaModel = .init(
        modelName: "HY-MT1.5-1.8B-Q4_K_M",
        stopTokens: StopToken.llama
    )
    
    static let llama3model: LLamaModel = .init(
        modelName: "Llama-3.2-3B-Instruct-Q3_K_L",
        stopTokens: StopToken.llama3
    )
    
    let currentModel = hyMTModel
    
    init() {
        let path = Bundle.main.path(forResource: currentModel.modelName, ofType: "gguf") ?? ""
        swiftLlama = (try? SwiftLlama(
            modelPath: path,
            modelConfiguration: .init(
                topK: Int(0.6),
                topP: 1.05,
                nCTX: 4096,
                temperature: 0.7,
            ))
        )!
    }
    
    func run(for userMessage: String) {
        result = ""
        
        let prompt = Prompt(systemPrompt: "",
                            userMessage: userMessage)
        Task {
            switch usingStream {
            case true:
                for try await value in await swiftLlama.start(for: prompt) {
                    result += value
                }
            case false:
                await swiftLlama.start(for: prompt)
                    .sink { _ in
                        
                    } receiveValue: {[weak self] value in
                        self?.result += value
                    }.store(in: &cancallable)
            }
        }
    }
}
