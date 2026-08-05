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
    let affine8PackedKeys: [UInt8]?
    let affine8PackedValues: [UInt8]?
    let sinks: [Float]
    let expected: [Float]
    let affine8Expected: [Float]?
    enum CodingKeys: String, CodingKey {
        case schemaVersion="schema_version", semantic, sourceRevision="source_revision", sourceSha256="source_sha256", format, context, qHeads="q_heads", kvHeads="kv_heads", rotatedQueries="rotated_queries_f32", packedKeys="packed_keys_u8", packedValues="packed_values_u8", affine8PackedKeys="wht_affine8_packed_keys_u8", affine8PackedValues="wht_affine8_packed_values_u8", sinks="sinks_f32", expected="expected_attention_f32", affine8Expected="wht_affine8_expected_attention_f32"
    }
}
struct Shape { var context, format, keyStride, valueStride, qHeads, kvHeads, useSinks: UInt32 }
func fail(_ message:String)->Never { FileHandle.standardError.write(Data("error: \(message)\n".utf8));exit(1) }
func relativeL2(_ actual:[Float],_ expected:[Float])->Double {
    let error=zip(actual,expected).reduce(0.0){$0+pow(Double($1.0-$1.1),2)},reference=expected.reduce(0.0){$0+pow(Double($1),2)};return sqrt(error/reference)
}
guard (4...6).contains(CommandLine.arguments.count) else { fail("usage: metal_mtp_attention_fixture <fixture.json> <kernel.metal> <locked-ggml-turbo-quant.c> [turbo4|wht_affine8] [output.f32]") }
let selectedFormat=CommandLine.arguments.count>=5 ? CommandLine.arguments[4] : "turbo4"
let outputPath=CommandLine.arguments.count==6 ? CommandLine.arguments[5] : nil
guard ["turbo4","wht_affine8"].contains(selectedFormat) else { fail("unknown learned fixture format") }
let fixture=try JSONDecoder().decode(Fixture.self,from:Data(contentsOf:URL(fileURLWithPath:CommandLine.arguments[1])))
guard fixture.schemaVersion==1,fixture.semantic=="mimo_mtp_real_attention_context17",fixture.sourceRevision=="63651580ca774f8504f676040460aed3e1244ac1",fixture.sourceSha256=="a0e41a193b2762b0c83e577f83206d0777028de6916408c8c368730c0c9e2143",fixture.format=="turbo4",fixture.context==17,fixture.qHeads==64,fixture.kvHeads==8,fixture.rotatedQueries.count==64*256,fixture.packedKeys.count==8*17*136,fixture.packedValues.count==8*17*68,fixture.sinks.count==64,fixture.expected.count==64*128 else { fail("fixture identity or shape mismatch") }
let selectedKeys:[UInt8],selectedValues:[UInt8],selectedExpected:[Float]
let formatValue:UInt32,keyStride:UInt32,valueStride:UInt32
if selectedFormat=="wht_affine8" {
    guard let keys=fixture.affine8PackedKeys,let values=fixture.affine8PackedValues,let expected=fixture.affine8Expected,keys.count==8*17*260,values.count==8*17*130,expected.count==64*128 else{fail("affine8 fixture identity or shape mismatch")}
    selectedKeys=keys;selectedValues=values;selectedExpected=expected;formatValue=8;keyStride=260;valueStride=130
} else {
    selectedKeys=fixture.packedKeys;selectedValues=fixture.packedValues;selectedExpected=fixture.expected;formatValue=4;keyStride=136;valueStride=68
}
let lockedSource=try String(contentsOfFile:CommandLine.arguments[3],encoding:.utf8)
func signs(_ name:String)->[Float] { let marker="turbo_cpu_\(name)[128] = {";guard let start=lockedSource.range(of:marker)?.upperBound,let end=lockedSource[start...].range(of:"};")?.lowerBound else{fail("locked signs missing")};let values=lockedSource[start..<end].split{$0==","||$0.isWhitespace}.compactMap{Float($0)};guard values.count==128 else{fail("locked signs invalid")};return values }
let s1=signs("s1"),s2=signs("s2")
guard let device=MTLCreateSystemDefaultDevice(),let queue=device.makeCommandQueue() else{fail("Metal unavailable")}
let source=try String(contentsOfFile:CommandLine.arguments[2],encoding:.utf8),library=try device.makeLibrary(source:source,options:nil)
guard let function=library.makeFunction(name:"turbo_gqa_attention_shared_kv") else{fail("shared KV kernel absent")}
let pipeline=try device.makeComputePipelineState(function:function)
func buffer<T>(_ values:[T])->MTLBuffer{values.withUnsafeBytes{device.makeBuffer(bytes:$0.baseAddress!,length:$0.count)!}}
let guarded=[Float](repeating:Float.nan,count:64*130);var shape=Shape(context:17,format:formatValue,keyStride:keyStride,valueStride:valueStride,qHeads:64,kvHeads:8,useSinks:1)
let kb=buffer(selectedKeys),vb=buffer(selectedValues),qb=buffer(fixture.rotatedQueries),ob=buffer(guarded),sb=device.makeBuffer(bytes:&shape,length:MemoryLayout<Shape>.stride)!,s1b=buffer(s1),s2b=buffer(s2),sinkb=buffer(fixture.sinks)
let command=queue.makeCommandBuffer()!,encoder=command.makeComputeCommandEncoder()!;encoder.setComputePipelineState(pipeline)
[kb,vb,qb,ob,sb,s1b,s2b,sinkb].enumerated().forEach{encoder.setBuffer($0.element,offset:0,index:$0.offset)}
encoder.dispatchThreadgroups(MTLSize(width:8,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:256,height:1,depth:1));encoder.endEncoding();command.commit();command.waitUntilCompleted();if let error=command.error{fail("Metal: \(error)")}
let pointer=ob.contents().bindMemory(to:Float.self,capacity:64*130);var actual=[Float](),guards=true
for head in 0..<64{guards=guards&&pointer[head*130].isNaN&&pointer[head*130+129].isNaN;actual+=Array(UnsafeBufferPointer(start:pointer+head*130+1,count:128))}
let rel=relativeL2(actual,selectedExpected),maxError=zip(actual,selectedExpected).map{abs($0-$1)}.max()!
guard guards,actual.allSatisfy({$0.isFinite}),rel<=4e-4,maxError<=7e-4 else{fail("fixture parity rel=\(rel) max=\(maxError) guards=\(guards)")}
if let outputPath {
    let bytes=actual.withUnsafeBytes{Data($0)}
    do { try bytes.write(to:URL(fileURLWithPath:outputPath),options:.withoutOverwriting) }
    catch { fail("cannot create Metal attention artifact: \(error)") }
}
let result:[String:Any]=["schema_version":1,"device":device.name,"semantic":fixture.semantic,"format":selectedFormat,"metal_vs_scalar_relative_l2":rel,"metal_vs_scalar_max_abs":maxError,"all_head_guards_intact":guards,"emitted_float_count":outputPath == nil ? 0 : actual.count]
let output=try JSONSerialization.data(withJSONObject:result,options:[.sortedKeys]);FileHandle.standardOutput.write(output);FileHandle.standardOutput.write(Data("\n".utf8))
