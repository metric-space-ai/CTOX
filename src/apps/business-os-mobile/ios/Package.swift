// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "CTOXBusinessOSMobile",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [.library(name: "CTOXMobileCore", targets: ["CTOXMobileCore"])],
    targets: [
        .target(name: "CTOXMobileCore"),
        .testTarget(name: "CTOXMobileCoreTests", dependencies: ["CTOXMobileCore"]),
    ]
)
