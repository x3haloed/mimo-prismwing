import Foundation
import Metal

struct Fixture: Decodable {
    let schemaVersion: Int
    let semantic: String
    let sourceRevision: String
    let sourceSha256: String
    let format: String
    let context: Int
    let qHeads: Int
    let kvHeads: Int
    let rotatedQueries: [Float]
    let packedKeys: [UInt8]
    let packedValues: [UInt8]
    let sinks: [Float]
    let expected: [Float]
    enum CodingKeys: String, CodingKey {
        case schemaVersion="schema_version", semantic, sourceRevision="source_revision", sourceSha256="source_sha256", format, context, qHeads="q_heads", kvHeads="kv_heads", rotatedQueries="rotated_queries_f32", packedKeys="packed_keys_u8", packedValues="packed_values_u8", sinks="sinks_f32", expected="expected_attention_f32"
    }
}
struct Shape { var context, format, keyStride, valueStride, qHeads, kvHeads, useSinks: UInt32 }
func fail(_ message:String)->Never { FileHandle.standardError.write(Data("error: \(message)\n".utf8));exit(1) }
func relativeL2(_ actual:[Float],_ expected:[Float])->Double {
    let error=zip(actual,expected).reduce(0.0){$0+pow(Double($1.0-$1.1),2)},reference=expected.reduce(0.0){$0+pow(Double($1),2)};return sqrt(error/reference)
}
guard CommandLine.arguments.count==4 else { fail("usage: metal_mtp_attention_fixture <fixture.json> <kernel.metal> <locked-ggml-turbo-quant.c>") }
let fixture=try JSONDecoder().decode(Fixture.self,from:Data(contentsOf:URL(fileURLWithPath:CommandLine.arguments[1])))
guard fixture.schemaVersion==1,fixture.semantic=="mimo_mtp_real_attention_context17",fixture.sourceRevision=="63651580ca774f8504f676040460aed3e1244ac1",fixture.sourceSha256=="a0e41a193b2762b0c83e577f83206d0777028de6916408c8c368730c0c9e2143",fixture.format=="turbo4",fixture.context==17,fixture.qHeads==64,fixture.kvHeads==8,fixture.rotatedQueries.count==64*256,fixture.packedKeys.count==8*17*136,fixture.packedValues.count==8*17*68,fixture.sinks.count==64,fixture.expected.count==64*128 else { fail("fixture identity or shape mismatch") }
let lockedSource=try String(contentsOfFile:CommandLine.arguments[3],encoding:.utf8)
func signs(_ name:String)->[Float] { let marker="turbo_cpu_\(name)[128] = {";guard let start=lockedSource.range(of:marker)?.upperBound,let end=lockedSource[start...].range(of:"};")?.lowerBound else{fail("locked signs missing")};let values=lockedSource[start..<end].split{$0==","||$0.isWhitespace}.compactMap{Float($0)};guard values.count==128 else{fail("locked signs invalid")};return values }
let s1=signs("s1"),s2=signs("s2")
guard let device=MTLCreateSystemDefaultDevice(),let queue=device.makeCommandQueue() else{fail("Metal unavailable")}
let source=try String(contentsOfFile:CommandLine.arguments[2],encoding:.utf8),library=try device.makeLibrary(source:source,options:nil)
guard let function=library.makeFunction(name:"turbo_gqa_attention_shared_kv") else{fail("shared KV kernel absent")}
let pipeline=try device.makeComputePipelineState(function:function)
func buffer<T>(_ values:[T])->MTLBuffer{values.withUnsafeBytes{device.makeBuffer(bytes:$0.baseAddress!,length:$0.count)!}}
let guarded=[Float](repeating:Float.nan,count:64*130);var shape=Shape(context:17,format:4,keyStride:136,valueStride:68,qHeads:64,kvHeads:8,useSinks:1)
let kb=buffer(fixture.packedKeys),vb=buffer(fixture.packedValues),qb=buffer(fixture.rotatedQueries),ob=buffer(guarded),sb=device.makeBuffer(bytes:&shape,length:MemoryLayout<Shape>.stride)!,s1b=buffer(s1),s2b=buffer(s2),sinkb=buffer(fixture.sinks)
let command=queue.makeCommandBuffer()!,encoder=command.makeComputeCommandEncoder()!;encoder.setComputePipelineState(pipeline)
[kb,vb,qb,ob,sb,s1b,s2b,sinkb].enumerated().forEach{encoder.setBuffer($0.element,offset:0,index:$0.offset)}
encoder.dispatchThreadgroups(MTLSize(width:8,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:256,height:1,depth:1));encoder.endEncoding();command.commit();command.waitUntilCompleted();if let error=command.error{fail("Metal: \(error)")}
let pointer=ob.contents().bindMemory(to:Float.self,capacity:64*130);var actual=[Float](),guards=true
for head in 0..<64{guards=guards&&pointer[head*130].isNaN&&pointer[head*130+129].isNaN;actual+=Array(UnsafeBufferPointer(start:pointer+head*130+1,count:128))}
let rel=relativeL2(actual,fixture.expected),maxError=zip(actual,fixture.expected).map{abs($0-$1)}.max()!
guard guards,actual.allSatisfy({$0.isFinite}),rel<=4e-4,maxError<=7e-4 else{fail("fixture parity rel=\(rel) max=\(maxError) guards=\(guards)")}
let result:[String:Any]=["schema_version":1,"device":device.name,"semantic":fixture.semantic,"metal_vs_scalar_relative_l2":rel,"metal_vs_scalar_max_abs":maxError,"all_head_guards_intact":guards]
let output=try JSONSerialization.data(withJSONObject:result,options:[.sortedKeys]);FileHandle.standardOutput.write(output);FileHandle.standardOutput.write(Data("\n".utf8))
