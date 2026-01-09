// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "SwiftLlama",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
        .watchOS(.v11),
        .tvOS(.v18),
        .visionOS(.v2)
    ],
    products: [
        .library(name: "SwiftLlama", targets: ["SwiftLlama"]),
    ],
    dependencies: [],
    targets: [
        .binaryTarget(
            name: "LlamaFramework",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b7681/llama-b7681-xcframework.zip",
            checksum: "2e7620b1eba6feb2c384cf791272bba14ff5a443a5640f892458f1e1310f7f04"
        ),
        .target(
            name: "SwiftLlama",
            dependencies: ["LlamaFramework"]
        ),
        .testTarget(
            name: "SwiftLlamaTests",
            dependencies: ["SwiftLlama"]
        )
    ]
)
