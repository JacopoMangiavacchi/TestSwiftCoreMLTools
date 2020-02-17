// swift-tools-version:5.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "CoreML-Training",
    products: [
        // Products define the executables and libraries produced by a package, and make them visible to other packages.
        .library(
            name: "CoreML-Training",
            targets: ["CoreML-Training"]),
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        // .package(url: /* package url */, from: "1.0.0"),
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages which this package depends on.
        .target(
            name: "CoreML-Training",
            dependencies: []),
        .testTarget(
            name: "CoreML-TrainingTests",
            dependencies: ["CoreML-Training"]),
    ]
)

package.dependencies = [
    .package(url: "https://github.com/JacopoMangiavacchi/SwiftCoreMLTools", from: "0.0.0")
]
package.targets = [.target(name: "CoreML-Training", dependencies: ["SwiftCoreMLTools"])]