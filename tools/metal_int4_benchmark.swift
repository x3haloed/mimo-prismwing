import Foundation
import Metal

struct Fixture: Decodable {
    let input: [Float]
    let expected: [Float]
    let packed: [[UInt8]]
    let scales: [[Float]]

    enum CodingKeys: String, CodingKey {
        case input
        case expected = "expected_f32"
        case packed = "packed_u8"
        case scales = "scale"
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

guard (5...7).contains(CommandLine.arguments.count) else {
    fail("usage: metal_int4_benchmark <model_mtp.safetensors> <fixture.json> <kernel.metal> <parallel-lanes> [batch] [scalar|vector]")
}
let checkpointURL = URL(fileURLWithPath: CommandLine.arguments[1])
let fixtureURL = URL(fileURLWithPath: CommandLine.arguments[2])
let kernelURL = URL(fileURLWithPath: CommandLine.arguments[3])
let requestedLanes = Int(CommandLine.arguments[4])
let batch = CommandLine.arguments.count >= 6 ? Int(CommandLine.arguments[5]) : 1
let gemm8Variant = CommandLine.arguments.count == 7 ? CommandLine.arguments[6] : "scalar"
guard let requestedLanes, requestedLanes > 0, requestedLanes <= 1024,
      requestedLanes.nonzeroBitCount == 1, let batch, batch == 1 || batch == 8,
      ["scalar", "vector"].contains(gemm8Variant),
      batch == 8 || gemm8Variant == "scalar" else {
    fail("invalid parallel width, batch, or batch-eight variant")
}
let weightName = "model.mtp.layers.0.mlp.gate_proj.weight"
let scaleName = "model.mtp.layers.0.mlp.gate_proj.weight_scale_inv"
let rows = 16_384
let columns = 4_096
let blockRows = 128
let blockColumns = 128
let scaleColumns = columns / blockColumns

let fixture: Fixture
do {
    fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: fixtureURL))
} catch {
    fail("fixture decode failed: \(error)")
}
guard fixture.input.count == columns, fixture.expected.count >= 4,
      fixture.packed.count >= 4, fixture.scales.count >= 4 else {
    fail("fixture has wrong dimensions")
}

guard let checkpoint = try? FileHandle(forReadingFrom: checkpointURL),
      let prefix = try? checkpoint.read(upToCount: 8), prefix.count == 8 else {
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

let weightOffsets = tensorMetadata(weightName, dtype: "F8_E4M3", shape: [rows, columns])
let scaleOffsets = tensorMetadata(
    scaleName,
    dtype: "F32",
    shape: [rows / blockRows, scaleColumns]
)
let payloadOffset = Int64(8 + headerLength)
do { try checkpoint.seek(toOffset: UInt64(payloadOffset + weightOffsets.0)) }
catch { fail("weight seek failed: \(error)") }
let sourceWeightBytes = Int(weightOffsets.1 - weightOffsets.0)
guard let sourceWeights = try? checkpoint.read(upToCount: sourceWeightBytes),
      sourceWeights.count == sourceWeightBytes else { fail("short weight read") }
do { try checkpoint.seek(toOffset: UInt64(payloadOffset + scaleOffsets.0)) }
catch { fail("scale seek failed: \(error)") }
let sourceScaleBytes = Int(scaleOffsets.1 - scaleOffsets.0)
guard let sourceScales = try? checkpoint.read(upToCount: sourceScaleBytes),
      sourceScales.count == sourceScaleBytes else { fail("short scale read") }
try? checkpoint.close()

let decodeTable = (0...255).map { decodeFP8(UInt8($0)) }
var packedWeights = [UInt8](repeating: 0, count: rows * columns / 2)
var quantScales = [Float](repeating: 0, count: rows * scaleColumns)
sourceWeights.withUnsafeBytes { sourceRaw in
    sourceScales.withUnsafeBytes { scaleRaw in
        let source = sourceRaw.bindMemory(to: UInt8.self)
        let scales = scaleRaw.bindMemory(to: Float.self)
        for row in 0..<rows {
            for block in 0..<scaleColumns {
                let sourceScale = scales[(row / blockRows) * scaleColumns + block]
                let columnBase = block * blockColumns
                var maximum: Float = 0
                for within in 0..<blockColumns {
                    let value = decodeTable[Int(source[row * columns + columnBase + within])] * sourceScale
                    maximum = max(maximum, abs(value))
                }
                guard maximum > 0 else { fail("zero quantization block is unsupported") }
                let quantScale = maximum / 7
                quantScales[row * scaleColumns + block] = quantScale
                for pair in 0..<(blockColumns / 2) {
                    let firstColumn = columnBase + pair * 2
                    let first = decodeTable[Int(source[row * columns + firstColumn])] * sourceScale
                    let second = decodeTable[Int(source[row * columns + firstColumn + 1])] * sourceScale
                    let low = max(-7, min(7, Int((first / quantScale).rounded(.toNearestOrEven))))
                    let high = max(-7, min(7, Int((second / quantScale).rounded(.toNearestOrEven))))
                    packedWeights[row * (columns / 2) + firstColumn / 2] =
                        (UInt8(bitPattern: Int8(low)) & 0x0f) |
                        ((UInt8(bitPattern: Int8(high)) & 0x0f) << 4)
                }
            }
        }
    }
}
for row in 0..<4 {
    let packedStart = row * columns / 2
    guard Array(packedWeights[packedStart..<(packedStart + columns / 2)]) == fixture.packed[row] else {
        fail("quantized bytes disagree with fixture at row \(row)")
    }
    let scaleStart = row * scaleColumns
    guard Array(quantScales[scaleStart..<(scaleStart + scaleColumns)]) == fixture.scales[row] else {
        fail("quantization scales disagree with fixture at row \(row)")
    }
}

guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    fail("Metal device unavailable")
}
let source: String
do { source = try String(contentsOf: kernelURL, encoding: .utf8) }
catch { fail("kernel source read failed: \(error)") }
let pipeline: MTLComputePipelineState
do {
    let library = try device.makeLibrary(source: source, options: nil)
    let functionName = if batch == 1 {
        "group_int4_gemv_parallel_blocked"
    } else if gemm8Variant == "vector" {
        "group_int4_gemm8_vector_parallel_blocked"
    } else {
        "group_int4_gemm8_parallel_blocked"
    }
    guard let function = library.makeFunction(name: functionName) else {
        fail("INT4 kernel function absent")
    }
    pipeline = try device.makeComputePipelineState(function: function)
} catch { fail("Metal compilation failed: \(error)") }
guard pipeline.maxTotalThreadsPerThreadgroup >= requestedLanes else {
    fail("device cannot dispatch required parallel width")
}

let batchedInput: [Float]
if batch == 1 {
    batchedInput = fixture.input
} else if gemm8Variant == "vector" {
    batchedInput = fixture.input.flatMap { value in Array(repeating: value, count: batch) }
} else {
    batchedInput = Array(repeating: fixture.input, count: batch).flatMap { $0 }
}
guard let weightBuffer = device.makeBuffer(bytes: packedWeights, length: packedWeights.count),
      let scaleBuffer = quantScales.withUnsafeBytes({
          device.makeBuffer(bytes: $0.baseAddress!, length: $0.count)
      }),
      let inputBuffer = batchedInput.withUnsafeBytes({
          device.makeBuffer(bytes: $0.baseAddress!, length: $0.count)
      }),
      let outputBuffer = device.makeBuffer(
        length: batch * rows * MemoryLayout<Float>.stride,
        options: .storageModeShared
      ) else { fail("Metal buffer allocation failed") }
var shape = GemvShape(
    rows: UInt32(rows), columns: UInt32(columns),
    blockRows: 1, blockColumns: UInt32(blockColumns)
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
    encoder.setThreadgroupMemoryLength(batch * requestedLanes * MemoryLayout<Float>.stride, index: 0)
    encoder.dispatchThreadgroups(
        MTLSize(width: rows, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: requestedLanes, height: 1, depth: 1)
    )
    encoder.endEncoding()
    let start = DispatchTime.now().uptimeNanoseconds
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let end = DispatchTime.now().uptimeNanoseconds
    if let error = commandBuffer.error { fail("Metal execution failed: \(error)") }
    return (
        Double(end - start) / 1_000_000.0,
        (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1_000.0
    )
}

for _ in 0..<5 { _ = run() }
var wall = [Double]()
var gpu = [Double]()
for _ in 0..<30 {
    let measurement = run()
    wall.append(measurement.0)
    gpu.append(measurement.1)
}
let output = outputBuffer.contents().bindMemory(to: Float.self, capacity: batch * rows)
var maxError: Float = 0
for item in 0..<batch {
    let firstFour = Array(UnsafeBufferPointer(start: output + item * rows, count: 4))
    maxError = max(maxError, zip(firstFour, fixture.expected).map { abs($0 - $1) }.max() ?? .infinity)
}
guard maxError < 2e-6 else { fail("full projection correctness mismatch: \(maxError)") }
let sortedWall = wall.sorted()
let sortedGpu = gpu.sorted()
let medianGpuMs = percentile(sortedGpu, 0.5)
let executableBytes = packedWeights.count + quantScales.count * MemoryLayout<Float>.stride
let result: [String: Any] = [
    "schema_version": 1,
    "device": device.name,
    "tensor": weightName,
    "kernel": batch == 1
        ? "group_int4_gemv_parallel_blocked"
        : (gemm8Variant == "vector"
            ? "group_int4_gemm8_vector_parallel_blocked"
            : "group_int4_gemm8_parallel_blocked"),
    "parallel_lanes": requestedLanes,
    "rows": rows,
    "columns": columns,
    "batch_size": batch,
    "concurrency": 1,
    "warmup_runs": 5,
    "measured_runs": 30,
    "cache_state": "application buffers warm; source load and quantization excluded",
    "packed_weight_bytes": packedWeights.count,
    "scale_bytes": quantScales.count * MemoryLayout<Float>.stride,
    "executable_bytes": executableBytes,
    "wall_ms": wall,
    "gpu_ms": gpu,
    "wall_median_ms": percentile(sortedWall, 0.5),
    "wall_p10_ms": percentile(sortedWall, 0.1),
    "wall_p90_ms": percentile(sortedWall, 0.9),
    "gpu_median_ms": medianGpuMs,
    "gpu_p10_ms": percentile(sortedGpu, 0.1),
    "gpu_p90_ms": percentile(sortedGpu, 0.9),
    "executable_gib_per_second_median": Double(executableBytes) / Double(1 << 30) / (medianGpuMs / 1_000),
    "effective_accepted_projection_tps": Double(batch) / (medianGpuMs / 1_000),
    "arithmetic_gflop_per_second": Double(2 * rows * columns * batch) / (medianGpuMs / 1_000) / 1_000_000_000,
    "source_fp8_logical_gib_per_second_median": Double(sourceWeightBytes) / Double(1 << 30) / (medianGpuMs / 1_000),
    "first_four_max_abs_error_vs_int4_oracle": maxError,
]
let resultData = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
FileHandle.standardOutput.write(resultData)
FileHandle.standardOutput.write(Data("\n".utf8))
