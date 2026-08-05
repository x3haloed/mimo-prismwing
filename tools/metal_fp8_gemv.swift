import Foundation
import Metal

struct Fixture: Decodable {
    let schemaVersion: Int
    let semantic: String
    let rows: Int
    let columns: Int
    let blockColumns: Int
    let raw: [[UInt8]]
    let scales: [Float]
    let input: [Float]
    let expected: [Float]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case semantic, rows, columns
        case blockColumns = "block_columns"
        case raw = "raw_u8"
        case scales = "scale_inv"
        case input
        case expected = "expected_f32"
    }
}

struct GemvShape {
    var rows: UInt32
    var columns: UInt32
    var blockColumns: UInt32
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("error: \(message)\n".utf8))
    exit(1)
}

guard CommandLine.arguments.count == 3 else {
    fail("usage: metal_fp8_gemv <fixture.json> <kernel.metal>")
}

let fixtureURL = URL(fileURLWithPath: CommandLine.arguments[1])
let kernelURL = URL(fileURLWithPath: CommandLine.arguments[2])
let fixture: Fixture
do {
    fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: fixtureURL))
} catch {
    fail("fixture decode failed: \(error)")
}
guard fixture.schemaVersion == 1,
      fixture.semantic == "mimo_block_fp8_gemv_slice",
      fixture.raw.count == fixture.rows,
      fixture.raw.allSatisfy({ $0.count == fixture.columns }),
      fixture.input.count == fixture.columns,
      fixture.scales.count == fixture.columns / fixture.blockColumns,
      fixture.expected.count == fixture.rows else {
    fail("fixture dimensions or semantic are invalid")
}

guard let device = MTLCreateSystemDefaultDevice(),
      let queue = device.makeCommandQueue() else {
    fail("Metal device or command queue is unavailable")
}
let source: String
do {
    source = try String(contentsOf: kernelURL, encoding: .utf8)
} catch {
    fail("kernel source read failed: \(error)")
}
let library: MTLLibrary
let pipeline: MTLComputePipelineState
do {
    library = try device.makeLibrary(source: source, options: nil)
    guard let function = library.makeFunction(name: "block_fp8_gemv") else {
        fail("kernel function is absent")
    }
    pipeline = try device.makeComputePipelineState(function: function)
} catch {
    fail("Metal compilation failed: \(error)")
}

let flattened = fixture.raw.flatMap { $0 }
var shape = GemvShape(
    rows: UInt32(fixture.rows),
    columns: UInt32(fixture.columns),
    blockColumns: UInt32(fixture.blockColumns)
)
guard let weightBuffer = device.makeBuffer(bytes: flattened, length: flattened.count),
      let scaleBuffer = device.makeBuffer(
        bytes: fixture.scales,
        length: fixture.scales.count * MemoryLayout<Float>.stride
      ),
      let inputBuffer = device.makeBuffer(
        bytes: fixture.input,
        length: fixture.input.count * MemoryLayout<Float>.stride
      ),
      let outputBuffer = device.makeBuffer(
        length: fixture.rows * MemoryLayout<Float>.stride,
        options: .storageModeShared
      ),
      let shapeBuffer = device.makeBuffer(bytes: &shape, length: MemoryLayout<GemvShape>.stride),
      let commandBuffer = queue.makeCommandBuffer(),
      let encoder = commandBuffer.makeComputeCommandEncoder() else {
    fail("Metal buffer or encoder allocation failed")
}

encoder.setComputePipelineState(pipeline)
encoder.setBuffer(weightBuffer, offset: 0, index: 0)
encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
encoder.setBuffer(inputBuffer, offset: 0, index: 2)
encoder.setBuffer(outputBuffer, offset: 0, index: 3)
encoder.setBuffer(shapeBuffer, offset: 0, index: 4)
encoder.dispatchThreads(
    MTLSize(width: fixture.rows, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: min(fixture.rows, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
)
encoder.endEncoding()
commandBuffer.commit()
commandBuffer.waitUntilCompleted()
if let error = commandBuffer.error {
    fail("Metal execution failed: \(error)")
}

let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: fixture.rows)
let actual = Array(UnsafeBufferPointer(start: pointer, count: fixture.rows))
for row in 0..<fixture.rows {
    let error = abs(actual[row] - fixture.expected[row])
    if error >= 2e-6 {
        fail("row \(row) mismatch: actual \(actual[row]), expected \(fixture.expected[row]), error \(error)")
    }
}
let result: [String: Any] = [
    "device": device.name,
    "rows": fixture.rows,
    "columns": fixture.columns,
    "actual": actual,
    "expected": fixture.expected,
    "max_abs_error": zip(actual, fixture.expected).map { abs($0 - $1) }.max() ?? 0.0,
]
let output = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
FileHandle.standardOutput.write(output)
FileHandle.standardOutput.write(Data("\n".utf8))
