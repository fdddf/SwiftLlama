import SwiftUI

struct ContentView: View {
    @State private var viewModel = ViewModel()
    @State private var prompt = ""

    var body: some View {
        VStack {
            Text("Swift Llama Demo").font(.title)
            Toggle(isOn: $viewModel.usingStream, label: {
                Text("Choice of AsyncStream or Combine Publisher, on for Stream, off for Publisher")
            })
            TextField("Prompt", text: $prompt, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .onSubmit {
                    viewModel.run(for: prompt)
                }
            Text(viewModel.result)
            Text("Tokens: \(viewModel.tokenCount)")
                .onChange(of: prompt) { _, newValue in
                    viewModel.updateTokenCount(for: newValue)
                }
            Spacer()
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
