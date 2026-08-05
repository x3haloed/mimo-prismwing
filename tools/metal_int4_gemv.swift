import Foundation
import Metal

struct Fixture: Decodable {
    let schemaVersion: Int
    let semantic: String
    let rows: Int
    let columns: Int
    let blockColumns: Int
    let packed: [[UInt8]]
    let scales: [[Float]]
    let input: [Float]
    let expected: [Float]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case semantic, rows, columns
        case blockColumns = "block_columns"
        case packed = "packed_u8"
        case scales = "scale"
        case input
        case expected = "expected_f32"
    }
}

struct GemvShape {
    var rows: UInt32
    var columns: UInt32
    var blockRows: UInt32
    var blockColumns: UInt32
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("error: \(message)\n".utf8))
    exit(1)
}

guard (3...4).contains(CommandLine.arguments.count) else {
    fail("usage: metal_int4_gemv <fixture.json> <kernel.metal> [parallel-lanes]")
}
let fixtureURL = URL(fileURLWithPath: CommandLine.arguments[1])
let kernelURL = URL(fileURLWithPath: CommandLine.arguments[2])
let requestedLanes = CommandLine.arguments.count == 4 ? Int(CommandLine.arguments[3]) : 64
guard let requestedLanes, requestedLanes > 0, requestedLanes.nonzeroBitCount == 1 else {
    fail("parallel width must be a positive power of two")
}

let fixture: Fixture
do {
    fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: fixtureURL))
} catch {
    fail("fixture decode failed: \(error)")
}
let scaleColumns = fixture.columns / fixture.blockColumns
guard fixture.schemaVersion == 1,
      fixture.semantic == "mimo_signed_int4_group128_gemv_slice",
      fixture.blockColumns > 0,
      fixture.columns.isMultiple(of: fixture.blockColumns),
      fixture.packed.count == fixture.rows,
      fixture.packed.allSatisfy({ $0.count * 2 == fixture.columns }),
      fixture.scales.count == fixture.rows,
      fixture.scales.allSatisfy({ $0.count == scaleColumns }),
      fixture.input.count == fixture.columns,
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
let pipeline: MTLComputePipelineState
do {
    let library = try device.makeLibrary(source: source, options: nil)
    guard let function = library.makeFunction(name: "group_int4_gemv_parallel_blocked") else {
        fail("INT4 kernel function is absent")
    }
    pipeline = try device.makeComputePipelineState(function: function)
} catch {
    fail("Metal compilation failed: \(error)")
}
guard pipeline.maxTotalThreadsPerThreadgroup >= requestedLanes else {
    fail("device cannot dispatch requested parallel width")
}

let flattenedWeights = fixture.packed.flatMap { $0 }
let flattenedScales = fixture.scales.flatMap { $0 }
var shape = GemvShape(
    rows: UInt32(fixture.rows),
    columns: UInt32(fixture.columns),
    blockRows: 1,
    blockColumns: UInt32(fixture.blockColumns)
)
guard let weightBuffer = device.makeBuffer(bytes: flattenedWeights, length: flattenedWeights.count),
      let scaleBuffer = device.makeBuffer(
        bytes: flattenedScales,
        length: flattenedScales.count * MemoryLayout<Float>.stride
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
encoder.setThreadgroupMemoryLength(requestedLanes * MemoryLayout<Float>.stride, index: 0)
encoder.dispatchThreadgroups(
    MTLSize(width: fixture.rows, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: requestedLanes, height: 1, depth: 1)
)
encoder.endEncoding()
commandBuffer.commit()
commandBuffer.waitUntilCompleted()
if let error = commandBuffer.error { fail("Metal execution failed: \(error)") }

let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: fixture.rows)
let actual = Array(UnsafeBufferPointer(start: pointer, count: fixture.rows))
let maxError = zip(actual, fixture.expected).map { abs($0 - $1) }.max() ?? .infinity
guard maxError < 2e-6 else { fail("INT4 fixture mismatch: \(maxError)") }
let result: [String: Any] = [
    "device": device.name,
    "kernel": "group_int4_gemv_parallel_blocked",
    "parallel_lanes": requestedLanes,
    "rows": fixture.rows,
    "columns": fixture.columns,
    "actual": actual,
    "expected": fixture.expected,
    "max_abs_error": maxError,
]
let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
FileHandle.standardOutput.write(data)
FileHandle.standardOutput.write(Data("\n".utf8))
