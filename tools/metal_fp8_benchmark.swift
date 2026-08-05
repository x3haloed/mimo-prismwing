import Foundation
import Metal

struct Fixture: Decodable {
    let input: [Float]
    let expected: [Float]

    enum CodingKeys: String, CodingKey {
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

func percentile(_ sorted: [Double], _ fraction: Double) -> Double {
    let index = Int((Double(sorted.count - 1) * fraction).rounded())
    return sorted[index]
}

guard (4...6).contains(CommandLine.arguments.count) else {
    fail("usage: metal_fp8_benchmark <model_mtp.safetensors> <fixture.json> <kernel.metal> [function] [parallel-lanes]")
}
let checkpointURL = URL(fileURLWithPath: CommandLine.arguments[1])
let fixtureURL = URL(fileURLWithPath: CommandLine.arguments[2])
let kernelURL = URL(fileURLWithPath: CommandLine.arguments[3])
let functionName = CommandLine.arguments.count >= 5 ? CommandLine.arguments[4] : "block_fp8_gemv"
let requestedLanes = CommandLine.arguments.count == 6 ? Int(CommandLine.arguments[5]) : 128
let parallelFunctions = [
    "block_fp8_gemv_parallel",
    "block_fp8_gemv_parallel_lut",
    "block_fp8_gemv_parallel_lut_blocked",
]
guard functionName == "block_fp8_gemv" || parallelFunctions.contains(functionName) else {
    fail("unsupported kernel function")
}
guard let requestedLanes, requestedLanes > 0, requestedLanes <= 1024,
      requestedLanes.nonzeroBitCount == 1 else {
    fail("parallel lanes must be a positive power of two")
}
let weightName = "model.mtp.layers.0.mlp.gate_proj.weight"
let scaleName = "model.mtp.layers.0.mlp.gate_proj.weight_scale_inv"

let fixture: Fixture
do {
    fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: fixtureURL))
} catch {
    fail("fixture decode failed: \(error)")
}

guard let checkpoint = try? FileHandle(forReadingFrom: checkpointURL),
      let prefix = try? checkpoint.read(upToCount: 8),
      prefix.count == 8 else {
    fail("cannot read safetensors prefix")
}
let headerLength = prefix.enumerated().reduce(UInt64(0)) { result, element in
    result | UInt64(element.element) << UInt64(element.offset * 8)
}
guard headerLength > 0, headerLength <= 256 * 1024 * 1024,
      let headerData = try? checkpoint.read(upToCount: Int(headerLength)),
      let header = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
    fail("invalid safetensors header")
}

func tensorMetadata(_ name: String, dtype: String, shape: [Int]) -> (Int64, Int64) {
    guard let value = header[name] as? [String: Any],
          value["dtype"] as? String == dtype,
          value["shape"] as? [Int] == shape,
          let offsets = value["data_offsets"] as? [Int64], offsets.count == 2,
          offsets[1] >= offsets[0] else {
        fail("unexpected metadata for \(name)")
    }
    return (offsets[0], offsets[1])
}

let rows = 16_384
let columns = 4_096
let blockRows = 128
let blockColumns = 128
let weightOffsets = tensorMetadata(weightName, dtype: "F8_E4M3", shape: [rows, columns])
let scaleOffsets = tensorMetadata(scaleName, dtype: "F32", shape: [rows / blockRows, columns / blockColumns])
let payloadOffset = Int64(8 + headerLength)
do {
    try checkpoint.seek(toOffset: UInt64(payloadOffset + weightOffsets.0))
} catch {
    fail("weight seek failed: \(error)")
}
let weightBytes = Int(weightOffsets.1 - weightOffsets.0)
guard let weights = try? checkpoint.read(upToCount: weightBytes), weights.count == weightBytes else {
    fail("short weight read")
}
do {
    try checkpoint.seek(toOffset: UInt64(payloadOffset + scaleOffsets.0))
} catch {
    fail("scale seek failed: \(error)")
}
let scaleBytes = Int(scaleOffsets.1 - scaleOffsets.0)
guard let scales = try? checkpoint.read(upToCount: scaleBytes), scales.count == scaleBytes else {
    fail("short scale read")
}
try? checkpoint.close()
guard fixture.input.count == columns, fixture.expected.count >= 4 else {
    fail("fixture has wrong dimensions")
}

guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    fail("Metal device unavailable")
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
    guard let function = library.makeFunction(name: functionName) else {
        fail("kernel function absent")
    }
    pipeline = try device.makeComputePipelineState(function: function)
} catch {
    fail("Metal compilation failed: \(error)")
}

let weightBuffer = weights.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: weights.count) }
let scaleBuffer = scales.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: scales.count) }
let inputBuffer = fixture.input.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count) }
func decodeFP8(_ value: UInt8) -> Float {
    let sign: Float = value & 0x80 == 0 ? 1 : -1
    let exponent = Int((value >> 3) & 0x0f)
    let mantissa = Int(value & 0x07)
    if exponent == 0 { return sign * pow(2, -6) * Float(mantissa) / 8 }
    if exponent == 15 && mantissa == 7 {
        return Float(bitPattern: UInt32(value & 0x80) << 24 | 0x7ff0_0000)
    }
    return sign * pow(2, Float(exponent - 7)) * (1 + Float(mantissa) / 8)
}
let decodeTable = (0...255).map { decodeFP8(UInt8($0)) }
let decodeBuffer = decodeTable.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count) }
guard let weightBuffer, let scaleBuffer, let inputBuffer,
      let decodeBuffer,
      let outputBuffer = device.makeBuffer(length: rows * MemoryLayout<Float>.stride, options: .storageModeShared) else {
    fail("Metal buffer allocation failed")
}
var shape = GemvShape(
    rows: UInt32(rows), columns: UInt32(columns),
    blockRows: UInt32(blockRows), blockColumns: UInt32(blockColumns)
)
guard let shapeBuffer = device.makeBuffer(bytes: &shape, length: MemoryLayout<GemvShape>.stride) else {
    fail("shape buffer allocation failed")
}

func run() -> (Double, Double) {
    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
        fail("command allocation failed")
    }
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(weightBuffer, offset: 0, index: 0)
    encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
    encoder.setBuffer(inputBuffer, offset: 0, index: 2)
    encoder.setBuffer(outputBuffer, offset: 0, index: 3)
    encoder.setBuffer(shapeBuffer, offset: 0, index: 4)
    if functionName == "block_fp8_gemv_parallel_lut" ||
        functionName == "block_fp8_gemv_parallel_lut_blocked" {
        encoder.setBuffer(decodeBuffer, offset: 0, index: 5)
    }
    if parallelFunctions.contains(functionName) {
        let lanes = requestedLanes
        guard pipeline.maxTotalThreadsPerThreadgroup >= lanes else {
            fail("device cannot dispatch required parallel width")
        }
        encoder.setThreadgroupMemoryLength(lanes * MemoryLayout<Float>.stride, index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: lanes, height: 1, depth: 1)
        )
    } else {
        encoder.dispatchThreads(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        )
    }
    encoder.endEncoding()
    let start = DispatchTime.now().uptimeNanoseconds
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let end = DispatchTime.now().uptimeNanoseconds
    if let error = commandBuffer.error { fail("Metal execution failed: \(error)") }
    return (Double(end - start) / 1_000_000.0, (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1_000.0)
}

for _ in 0..<5 { _ = run() }
var wall = [Double]()
var gpu = [Double]()
for _ in 0..<30 {
    let measurement = run()
    wall.append(measurement.0)
    gpu.append(measurement.1)
}
let output = outputBuffer.contents().bindMemory(to: Float.self, capacity: rows)
let firstFour = Array(UnsafeBufferPointer(start: output, count: 4))
let maxError = zip(firstFour, fixture.expected).map { abs($0 - $1) }.max() ?? .infinity
guard maxError < 2e-6 else { fail("full projection correctness mismatch: \(maxError)") }
let sortedWall = wall.sorted()
let sortedGpu = gpu.sorted()
let medianGpuMs = percentile(sortedGpu, 0.5)
let result: [String: Any] = [
    "schema_version": 1,
    "device": device.name,
    "tensor": weightName,
    "kernel": functionName,
    "parallel_lanes": parallelFunctions.contains(functionName) ? requestedLanes : 1,
    "rows": rows,
    "columns": columns,
    "batch_size": 1,
    "concurrency": 1,
    "warmup_runs": 5,
    "measured_runs": 30,
    "cache_state": "application buffers warm; source-file load excluded",
    "weight_bytes": weightBytes,
    "scale_bytes": scaleBytes,
    "wall_ms": wall,
    "gpu_ms": gpu,
    "wall_median_ms": percentile(sortedWall, 0.5),
    "wall_p10_ms": percentile(sortedWall, 0.1),
    "wall_p90_ms": percentile(sortedWall, 0.9),
    "gpu_median_ms": medianGpuMs,
    "gpu_p10_ms": percentile(sortedGpu, 0.1),
    "gpu_p90_ms": percentile(sortedGpu, 0.9),
    "logical_weight_gib_per_second_median": Double(weightBytes) / Double(1 << 30) / (medianGpuMs / 1_000.0),
    "first_four_max_abs_error": maxError,
]
let resultData = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
FileHandle.standardOutput.write(resultData)
FileHandle.standardOutput.write(Data("\n".utf8))
