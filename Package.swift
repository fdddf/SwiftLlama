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
    dependencies: [
        .package(url: "https://github.com/ggml-org/llama.cpp.git", revision: "5b8844ae531d8ff09c1c00a2022293d5b674c787")
    ],
    targets: [
        .target(name: "SwiftLlama", 
                dependencies: [
                    "LlamaFramework",
                    .product(name: "llama", package: "llama.cpp")
                ]),
        .testTarget(name: "SwiftLlamaTests", dependencies: ["SwiftLlama"]),
        .binaryTarget(
            name: "LlamaFramework",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b7664/llama-b7664-xcframework.zip",
            checksum: "b658df013c0bb6750203a09aa282d4a85e2f06c36edd72169aeca674326e690a"
        )
    ]
)
