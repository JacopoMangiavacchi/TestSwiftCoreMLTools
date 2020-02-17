import XCTest
@testable import CoreML_Training

final class CoreML_TrainingTests: XCTestCase {
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        XCTAssertEqual(CoreML_Training().text, "Hello, World!")
    }

    static var allTests = [
        ("testExample", testExample),
    ]
}
