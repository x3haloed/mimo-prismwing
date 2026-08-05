import Foundation
import Metal

struct Shape { var context, format, keyStride, valueStride, qHeads, kvHeads: UInt32 }

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("error: \(message)\n".utf8)); exit(1)
}
func percentile(_ values: [Double], _ fraction: Double) -> Double {
    let sorted = values.sorted(); return sorted[Int((Double(sorted.count - 1) * fraction).rounded())]
}
func signs(_ name: String, source: String) -> [Float] {
    guard let start = source.range(of: "turbo_cpu_\(name)[128] = {")?.upperBound,
          let end = source[start...].range(of: "};")?.lowerBound else { fail("missing locked WHT \(name)") }
    let result = source[start..<end].split { $0 == "," || $0.isWhitespace }.compactMap { Float($0) }
    guard result.count == 128, result.allSatisfy({ $0 == -1 || $0 == 1 }) else { fail("invalid WHT \(name)") }
    return result
}
func fwht(_ values: inout [Float], _ base: Int) {
    var width = 1
    while width < 128 {
        var group = 0
        while group < 128 {
            for offset in 0..<width {
                let a = values[base + group + offset], b = values[base + group + offset + width]
                values[base + group + offset] = a + b
                values[base + group + offset + width] = a - b
            }
            group += 2 * width
        }
        width *= 2
    }
    for index in 0..<128 { values[base + index] *= 0.08838834764831845 }
}
func rotate(_ values: inout [Float], inverse: Bool, s1: [Float], s2: [Float]) {
    for base in stride(from: 0, to: values.count, by: 128) {
        let first = inverse ? s2 : s1, second = inverse ? s1 : s2
        for index in 0..<128 { values[base + index] *= first[index] }
        fwht(&values, base)
        for index in 0..<128 { values[base + index] *= second[index] }
    }
}
let c3: [Float] = [-0.190685,-0.117832,-0.065717,-0.021460,0.021460,0.065717,0.117832,0.190685]
let c4: [Float] = [-0.173926,-0.117195,-0.089527,-0.068756,-0.051262,-0.035597,-0.020989,-0.006938,0.006938,0.020989,0.035597,0.051262,0.068756,0.089527,0.117195,0.173926]
func halfBytes(_ value: Float) -> [UInt8] {
    let bits = Float16(value).bitPattern
    return [UInt8(bits & 255), UInt8(bits >> 8)]
}
func quantize(_ input: [Float], format: Int, s1: [Float], s2: [Float]) -> [UInt8] {
    var result = [UInt8]()
    for base in stride(from: 0, to: input.count, by: 128) {
        let norm = sqrt(input[base..<(base + 128)].reduce(Float(0)) { $0 + $1 * $1 })
        var rotated = Array(input[base..<(base + 128)]).map { norm > 1e-10 ? $0 / norm : 0 }
        rotate(&rotated, inverse: false, s1: s1, s2: s2)
        let centroids = format == 3 ? c3 : c4
        let indices = rotated.map { value -> Int in
            if format == 3 {
                let thresholds: [Float] = [-0.154259,-0.091775,-0.043589,0,0.043589,0.091775,0.154259]
                return thresholds.firstIndex(where: { value < $0 }) ?? 7
            }
            let thresholds: [Float] = [-0.145560,-0.103361,-0.079142,-0.060009,-0.043430,-0.028293,-0.013963,0,0.013963,0.028293,0.043430,0.060009,0.079142,0.103361,0.145560]
            return thresholds.firstIndex(where: { value < $0 }) ?? 15
        }
        let recon = sqrt(indices.reduce(Float(0)) { $0 + centroids[$1] * centroids[$1] })
        result += halfBytes(recon > 1e-10 ? norm / recon : norm)
        if format == 3 {
            var low = [UInt8](repeating: 0, count: 32), high = [UInt8](repeating: 0, count: 16)
            for i in 0..<128 { low[i/4] |= UInt8(indices[i] & 3) << UInt8((i%4)*2); high[i/8] |= UInt8((indices[i] >> 2) & 1) << UInt8(i%8) }
            result += low; result += high
        } else {
            result += [0, 0]
            var packed = [UInt8](repeating: 0, count: 64)
            for i in 0..<128 { packed[i/2] |= UInt8(indices[i]) << UInt8((i%2)*4) }
            result += packed
        }
    }
    return result
}
func dequant(_ row: ArraySlice<UInt8>, format: Int, count: Int) -> [Float] {
    let bytes = Array(row), blockSize = format == 3 ? 50 : 68, centroids = format == 3 ? c3 : c4
    return (0..<count).map { column in
        let block = column / 128, within = column % 128, offset = block * blockSize
        let norm = Float(Float16(bitPattern: UInt16(bytes[offset]) | UInt16(bytes[offset+1]) << 8))
        let index: Int
        if format == 3 {
            let low = Int((bytes[offset+2+within/4] >> UInt8((within%4)*2)) & 3)
            let high = Int((bytes[offset+34+within/8] >> UInt8(within%8)) & 1)
            index = low | high << 2
        } else { index = Int((bytes[offset+4+within/2] >> UInt8((within%2)*4)) & 15) }
        return centroids[index] * norm
    }
}
func relativeL2(_ actual: [Float], _ expected: [Float]) -> Double {
    let error = zip(actual, expected).reduce(0.0) { $0 + pow(Double($1.0-$1.1),2) }
    let reference = expected.reduce(0.0) { $0 + pow(Double($1),2) }
    return sqrt(error/reference)
}

guard (5...6).contains(CommandLine.arguments.count), let format = Int(CommandLine.arguments[3]), [3,4].contains(format),
      let context = Int(CommandLine.arguments[4]), [17,128,1024,8192].contains(context) else {
    fail("usage: metal_turbo_attention <kernel.metal> <locked-ggml-turbo-quant.c> <3|4> <17|128|1024|8192> [serial|parallel32]")
}
let variant = CommandLine.arguments.count == 6 ? CommandLine.arguments[5] : "serial"
guard ["serial", "parallel32", "gqa4", "gqa8"].contains(variant) else { fail("unknown kernel variant") }
let kernelSource = try String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8)
let lockedSource = try String(contentsOfFile: CommandLine.arguments[2], encoding: .utf8)
let s1 = signs("s1", source: lockedSource), s2 = signs("s2", source: lockedSource)
var state: UInt64 = 0x91e10da5c79e7b1d
func nextValue() -> Float { state = state &* 6364136223846793005 &+ 1442695040888963407; return (Float(UInt32(state >> 32) & 0xffff)/32767.5-1)*1.75 }

if variant == "gqa4" || variant == "gqa8" {
    let qHeads = 64, kvHeads = variant == "gqa4" ? 4 : 8
    if variant == "gqa8" && context != 128 { fail("SWA GQA requires context 128") }
    let keyStride = format == 3 ? 100 : 136, valueStride = format == 3 ? 50 : 68
    var queries = [Float](repeating: 0, count: qHeads * 256)
    for head in 0..<qHeads { for column in 0..<192 { queries[head*256+column] = nextValue() } }
    var packedKeys = [UInt8](), packedValues = [UInt8]()
    var decodedKeys = Array(repeating: [[Float]](), count: kvHeads)
    var decodedValues = Array(repeating: [[Float]](), count: kvHeads)
    for head in 0..<kvHeads {
        for _ in 0..<context {
            var key = [Float](repeating: 0, count: 256), value = [Float](repeating: 0, count: 128)
            for column in 0..<192 { key[column] = nextValue() }
            for column in 0..<128 { value[column] = nextValue() }
            let keyPacked = quantize(key, format: format, s1: s1, s2: s2)
            let valuePacked = quantize(value, format: format, s1: s1, s2: s2)
            packedKeys += keyPacked; packedValues += valuePacked
            decodedKeys[head].append(dequant(keyPacked[...], format: format, count: 256))
            decodedValues[head].append(dequant(valuePacked[...], format: format, count: 128))
        }
    }
    var expected = [Float](); expected.reserveCapacity(qHeads * 128)
    for qHead in 0..<qHeads {
        let kvHead = qHead / (qHeads / kvHeads)
        var query = Array(queries[(qHead*256)..<((qHead+1)*256)])
        rotate(&query, inverse: false, s1: s1, s2: s2)
        var maximum = -Float.infinity, denominator: Float = 0, accumulator = [Float](repeating: 0, count: 128)
        for token in 0..<context {
            let score = zip(query, decodedKeys[kvHead][token]).reduce(Float(0)) { $0+$1.0*$1.1 } / sqrt(192)
            let nextMaximum=max(maximum,score), oldScale=maximum.isInfinite ? 0 : exp(maximum-nextMaximum), newScale=exp(score-nextMaximum)
            denominator=denominator*oldScale+newScale
            for column in 0..<128 { accumulator[column]=accumulator[column]*oldScale+newScale*decodedValues[kvHead][token][column] }
            maximum=nextMaximum
        }
        for column in 0..<128 { accumulator[column] /= denominator }
        rotate(&accumulator, inverse: true, s1: s1, s2: s2); expected += accumulator
    }
    guard let device=MTLCreateSystemDefaultDevice(), let queue=device.makeCommandQueue() else { fail("Metal unavailable") }
    let compileStart=DispatchTime.now().uptimeNanoseconds, library=try device.makeLibrary(source:kernelSource,options:nil)
    guard let function=library.makeFunction(name:"turbo_gqa_attention_256_128_parallel32") else { fail("GQA kernel absent") }
    let pipeline=try device.makeComputePipelineState(function:function), compileMs=Double(DispatchTime.now().uptimeNanoseconds-compileStart)/1e6
    let guarded=[Float](repeating:Float.nan,count:qHeads*130)
    var shape=Shape(context:UInt32(context),format:UInt32(format),keyStride:UInt32(keyStride),valueStride:UInt32(valueStride),qHeads:UInt32(qHeads),kvHeads:UInt32(kvHeads))
    func makeBuffer<T>(_ values:[T])->MTLBuffer { values.withUnsafeBytes { device.makeBuffer(bytes:$0.baseAddress!,length:$0.count)! } }
    let kb=makeBuffer(packedKeys),vb=makeBuffer(packedValues),qb=makeBuffer(queries),ob=makeBuffer(guarded),sb=device.makeBuffer(bytes:&shape,length:MemoryLayout<Shape>.stride)!,s1b=makeBuffer(s1),s2b=makeBuffer(s2)
    func runGQA()->(Double,Double) {
        let command=queue.makeCommandBuffer()!,encoder=command.makeComputeCommandEncoder()!;encoder.setComputePipelineState(pipeline)
        [kb,vb,qb,ob,sb,s1b,s2b].enumerated().forEach { encoder.setBuffer($0.element,offset:0,index:$0.offset) }
        encoder.dispatchThreadgroups(MTLSize(width:qHeads,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:32,height:1,depth:1));encoder.endEncoding()
        let start=DispatchTime.now().uptimeNanoseconds;command.commit();command.waitUntilCompleted();if let error=command.error { fail("Metal: \(error)") }
        return (Double(DispatchTime.now().uptimeNanoseconds-start)/1e6,(command.gpuEndTime-command.gpuStartTime)*1000)
    }
    let cold=runGQA();for _ in 0..<10{_=runGQA()};var walls=[Double](),gpus=[Double]();for _ in 0..<30{let value=runGQA();walls.append(value.0);gpus.append(value.1)}
    let pointer=ob.contents().bindMemory(to:Float.self,capacity:qHeads*130);var actual=[Float](),guards=true
    for head in 0..<qHeads { guards = guards && pointer[head*130].isNaN && pointer[head*130+129].isNaN; actual += Array(UnsafeBufferPointer(start:pointer+head*130+1,count:128)) }
    let maxError=zip(actual,expected).map{abs($0-$1)}.max()!,rel=relativeL2(actual,expected)
    guard guards,actual.allSatisfy({$0.isFinite}),rel<=3e-4,maxError<=5e-4 else { fail("GQA parity rel=\(rel) max=\(maxError) guards=\(guards)") }
    let result:[String:Any]=["schema_version":1,"device":device.name,"format":"turbo\(format)","mode":variant=="gqa4" ? "global" : "swa","q_heads":qHeads,"kv_heads":kvHeads,"context":context,"batch_size":1,"concurrency":1,"accepted_tokens":1,"A":"not_applicable","U":"not_applicable","bytes_read":packedKeys.count+packedValues.count+queries.count*4,"compile_ms":compileMs,"cold_wall_ms":cold.0,"cold_gpu_ms":cold.1,"warmups":10,"measurements":30,"wall_median_ms":percentile(walls,0.5),"wall_p95_ms":percentile(walls,0.95),"gpu_median_ms":percentile(gpus,0.5),"gpu_p95_ms":percentile(gpus,0.95),"metal_vs_scalar_relative_l2":rel,"metal_vs_scalar_max_abs":maxError,"all_head_guards_intact":guards,"cache_state":"packed application buffers warm; no model or storage I/O"]
    let data=try JSONSerialization.data(withJSONObject:result,options:[.sortedKeys]);FileHandle.standardOutput.write(data);FileHandle.standardOutput.write(Data("\n".utf8));exit(0)
}
var query = [Float](repeating: 0, count: 256)
for i in 0..<192 { query[i] = nextValue() }
var rawKeys = [[Float]](), rawValues = [[Float]](), packedKeys = [UInt8](), packedValues = [UInt8]()
for _ in 0..<context {
    var key = [Float](repeating: 0, count: 256), value = [Float](repeating: 0, count: 128)
    for i in 0..<192 { key[i] = nextValue() }; for i in 0..<128 { value[i] = nextValue() }
    rawKeys.append(key); rawValues.append(value)
    packedKeys += quantize(key, format: format, s1: s1, s2: s2); packedValues += quantize(value, format: format, s1: s1, s2: s2)
}
let keyStride = format == 3 ? 100 : 136, valueStride = format == 3 ? 50 : 68
var qrot = query; rotate(&qrot, inverse: false, s1: s1, s2: s2)
var maximum = -Float.infinity, denominator: Float = 0, scalarRotated = [Float](repeating: 0, count: 128)
var baselineMaximum = -Float.infinity, baselineDenominator: Float = 0, baselineOutput = [Float](repeating: 0, count: 128)
var baselineScores = [Float](), candidateScores = [Float]()
for token in 0..<context {
    let baseline = zip(query.prefix(192), rawKeys[token].prefix(192)).reduce(Float(0)) { $0 + $1.0*$1.1 } / sqrt(192)
    let key = dequant(packedKeys[(token*keyStride)..<((token+1)*keyStride)], format: format, count: 256)
    let value = dequant(packedValues[(token*valueStride)..<((token+1)*valueStride)], format: format, count: 128)
    let score = zip(qrot,key).reduce(Float(0)) { $0+$1.0*$1.1 } / sqrt(192)
    baselineScores.append(baseline); candidateScores.append(score)
    let nextBaselineMaximum = max(baselineMaximum,baseline), oldBaselineScale = baselineMaximum.isInfinite ? 0 : exp(baselineMaximum-nextBaselineMaximum), newBaselineScale = exp(baseline-nextBaselineMaximum)
    baselineDenominator = baselineDenominator*oldBaselineScale+newBaselineScale
    for i in 0..<128 { baselineOutput[i] = baselineOutput[i]*oldBaselineScale+newBaselineScale*rawValues[token][i] }
    baselineMaximum = nextBaselineMaximum
    let nextMaximum = max(maximum,score), oldScale = maximum.isInfinite ? 0 : exp(maximum-nextMaximum), newScale = exp(score-nextMaximum)
    denominator = denominator*oldScale+newScale
    for i in 0..<128 { scalarRotated[i] = scalarRotated[i]*oldScale+newScale*value[i] }
    maximum = nextMaximum
}
for i in 0..<128 { scalarRotated[i] /= denominator; baselineOutput[i] /= baselineDenominator }; var scalar = scalarRotated; rotate(&scalar, inverse: true, s1: s1, s2: s2)

guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { fail("Metal unavailable") }
let compileStart = DispatchTime.now().uptimeNanoseconds
let library = try device.makeLibrary(source: kernelSource, options: nil)
let functionName = variant == "serial" ? "turbo_attention_256_128" : "turbo_attention_256_128_parallel32"
guard let function = library.makeFunction(name: functionName) else { fail("kernel absent") }
let pipeline = try device.makeComputePipelineState(function: function)
let compileMs = Double(DispatchTime.now().uptimeNanoseconds-compileStart)/1e6
var guarded = [Float](repeating: Float.nan, count: 130), shape = Shape(context: UInt32(context), format: UInt32(format), keyStride: UInt32(keyStride), valueStride: UInt32(valueStride), qHeads: 1, kvHeads: 1)
func buffer<T>(_ values: [T]) -> MTLBuffer { values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count)! } }
let kb=buffer(packedKeys), vb=buffer(packedValues), qb=buffer(query), ob=buffer(guarded), sb=device.makeBuffer(bytes:&shape,length:MemoryLayout<Shape>.stride)!, s1b=buffer(s1), s2b=buffer(s2)
func run() -> (Double,Double) {
    let command=queue.makeCommandBuffer()!, encoder=command.makeComputeCommandEncoder()!; encoder.setComputePipelineState(pipeline)
    [kb,vb,qb,ob,sb,s1b,s2b].enumerated().forEach { encoder.setBuffer($0.element,offset:0,index:$0.offset) }
    let lanes = variant == "serial" ? 1 : 32
    encoder.dispatchThreads(MTLSize(width:lanes,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:lanes,height:1,depth:1)); encoder.endEncoding()
    let start=DispatchTime.now().uptimeNanoseconds; command.commit(); command.waitUntilCompleted(); if let e=command.error { fail("Metal: \(e)") }
    return (Double(DispatchTime.now().uptimeNanoseconds-start)/1e6,(command.gpuEndTime-command.gpuStartTime)*1000)
}
let cold=run(); for _ in 0..<10 { _=run() }; var walls=[Double](), gpus=[Double](); for _ in 0..<50 { let x=run(); walls.append(x.0); gpus.append(x.1) }
let pointer=ob.contents().bindMemory(to:Float.self,capacity:130), actual=Array(UnsafeBufferPointer(start:pointer+1,count:128))
guard pointer[0].isNaN && pointer[129].isNaN else { fail("output guard changed") }
let maxError=zip(actual,scalar).map { abs($0-$1) }.max()!, rel=relativeL2(actual,scalar)
guard actual.allSatisfy({$0.isFinite}), rel <= 1e-4, maxError <= 2e-4 else { fail("parity rel=\(rel) max=\(maxError)") }
let result:[String:Any] = ["schema_version":1,"device":device.name,"format":"turbo\(format)","kernel_variant":variant,"lanes":variant == "serial" ? 1 : 32,"context":context,"batch_size":1,"concurrency":1,"accepted_tokens":1,"A":"not_applicable","U":"not_applicable","bytes_read":packedKeys.count+packedValues.count+query.count*4,"compile_ms":compileMs,"cold_wall_ms":cold.0,"cold_gpu_ms":cold.1,"warmups":10,"measurements":50,"wall_median_ms":percentile(walls,0.5),"wall_p95_ms":percentile(walls,0.95),"gpu_median_ms":percentile(gpus,0.5),"gpu_p95_ms":percentile(gpus,0.95),"metal_vs_scalar_relative_l2":rel,"metal_vs_scalar_max_abs":maxError,"score_relative_l2_vs_fp32":relativeL2(candidateScores,baselineScores),"output_relative_l2_vs_fp32":relativeL2(scalar,baselineOutput),"guard_intact":true,"cache_state":"packed application buffers warm; no model or storage I/O"]
let data=try JSONSerialization.data(withJSONObject:result,options:[.sortedKeys]); FileHandle.standardOutput.write(data); FileHandle.standardOutput.write(Data("\n".utf8))
