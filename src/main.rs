#[cfg(target_os = "macos")]
use prismwing::{
    RealAttentionMoeRequest, RealBaseLayerRequest, benchmark_layer4_metal_native_transaction,
    benchmark_layer4_metal_ready_artifact, benchmark_layer4_two_barrier_transaction,
    benchmark_metal_io_acquisition, benchmark_pread_expert_acquisition,
    build_layer4_metal_ready_artifact, run_arbitrary_text_generation,
    run_arbitrary_text_route_trace, run_bounded_metal_routed_row, run_layer4_metal_diagnostic,
    run_metal_base_layer_attention, run_metal_checkpoint_offset_probe, run_metal_direct_fp8_expert,
    run_metal_direct_fp8_expert_batch8_shared_weight, run_metal_direct_mapped_fp8_gemv,
    run_metal_direct_route_replay_fp8_moe_block, run_metal_direct_source_bf16_fp8_gemv,
    run_metal_direct_source_bf16_fp8_gemv_audit,
    run_metal_direct_source_bf16_reduction_width_fp8_moe_block,
    run_metal_direct_source_bf16_route_replay_fp8_moe_block,
    run_metal_direct_source_bf16_silu_lut_route_replay_fp8_moe_block,
    run_metal_dynamic_fp8_moe_block, run_metal_dynamic_real_attention_fp8_moe_block,
    run_metal_fp8_expert, run_metal_fp8_expert_batch8, run_metal_fp8_expert_batch8_shared_weight,
    run_metal_fp8_moe_block, run_metal_fused_gate_up_fp8_moe_block,
    run_metal_incremental_text_endpoint, run_metal_mapped_fp8_gemv,
    run_metal_native_distribution_probe, run_metal_noaux_tc_router, run_metal_real_base_layer,
    run_metal_simdgroup_matrix_fp8_moe_block, run_metal_union_parallel_fp8_moe_block,
    run_pressure_residency_smoke, run_pressure_resident_checkpoint_pilot,
    run_staged_metal_fp8_expert, run_weight_install_tomography,
    run_wide_metal_jacobi_text_endpoint,
};
use prismwing::{
    build_census, inspect_mapped_tensor, repack_expert_container, run_mapped_fp8_gemv,
    verify_expert_container, write_census,
};
#[cfg(target_os = "macos")]
use prismwing::{
    run_full_prefix_trace, run_global_attention_capture_smoke, run_global_attention_sparsity_trace,
    run_prefill_route_coverage_trace, run_real_layer0_trace, run_real_layer1_expert_trace,
    run_real_layer1_routing_trace, run_real_layer2_trace, run_real_layer4_trace,
    run_real_layer7_trace, run_real_routed_layer_trace, run_route_only_trace,
    run_routed_mixture_activation_corpus, run_slow_text_endpoint,
    run_structured_sparse_layer0_trace,
};
use std::fs::OpenOptions;
use std::path::PathBuf;
use tokenizers::Tokenizer;

fn usage() -> ! {
    eprintln!("usage:");
    eprintln!("  prismwing census <model.safetensors.index.json> <checkpoint-dir> <output.json>");
    eprintln!("  prismwing repack <source.safetensors> <output.pwexpert> <tensor> [tensor ...]");
    eprintln!("  prismwing verify-container <container.pwexpert>");
    eprintln!("  prismwing inspect-tensor <source.safetensors> <tensor>");
    eprintln!("  prismwing tokenize <tokenizer.json> <utf8-text>");
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing route-only-trace <checkpoint-dir> <model.lock.json> <checkpoint-verification.json> <fixture.json> <output-dir> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing prefill-route-coverage-trace <checkpoint-dir> <model.lock.json> <checkpoint-verification.json> <fixture.json> <positions> <output-dir> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing global-attention-sparsity-trace <checkpoint-dir> <model.lock.json> <checkpoint-verification.json> <fixture.json> <pw0157-prefix512-manifest.json> <output-dir> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing structured-sparse-layer0-trace <checkpoint-dir> <model.lock.json> <checkpoint-verification.json> <authority-fixture.json> <pw0176-fixture-manifest.json> <output-dir> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing global-attention-capture-smoke <checkpoint-dir> <model.lock.json> <checkpoint-verification.json> <fixture.json> <prefix64-control-manifest.json> <output-dir> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing routed-mixture-activation-corpus <checkpoint-dir> <model.lock.json> <checkpoint-verification.json> <fixture.json> <pw0112-manifest.json> <output-dir> <commit>"
    );
    eprintln!(
        "  prismwing fp8-gemv <source.safetensors> <weight> <scale> <input.f32> <output.f32>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing metal-fp8-gemv <source.safetensors> <kernel.metal> <weight> <scale> <input.f32> <reference.f32> <output.f32>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing metal-direct-fp8-gemv <source.safetensors> <kernel.metal> <weight> <scale> <input.f32> <reference.f32> <output.f32> <report.json>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing metal-direct-source-bf16-fp8-gemv <source.safetensors> <kernel.metal> <weight> <scale> <input.f32> <reference.f32> <output.f32> <report.json>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing metal-direct-source-bf16-fp8-gemv-audit <source.safetensors> <kernel.metal> <weight> <scale> <input.f32> <reference.f32> <output.f32> <report.json>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing metal-checkpoint-offset-probe <source.safetensors> <tensor> <output.json>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing metal-base-layer-attention <source.safetensors> <kernel.metal> <manifest.json> <input.f32> <output.f32>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing metal-fp8-expert <gate-up.safetensors> <down.safetensors> <kernel.metal> <input.f32> <reference.f32> <output.f32>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing metal-direct-fp8-expert <gate-up-shard.safetensors> <down-shard.safetensors> <kernel.metal> <input.f32> <reference.f32> <output.f32> <report.json>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing metal-direct-fp8-expert-batch8-shared-weight <gate-up-shard.safetensors> <down-shard.safetensors> <kernel.metal> <input.f32> <reference.f32> <output.f32> <report.json>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing metal-direct-route-replay-fp8-moe-block <manifest.json> <checkpoint-dir> <checkpoint-verification.json> <kernel.metal> <input.f32> <reference.f32> <output.f32> <report.json>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing metal-direct-source-bf16-route-replay-fp8-moe-block <manifest.json> <checkpoint-dir> <checkpoint-verification.json> <kernel.metal> <input.f32> <reference.f32> <output.f32> <report.json>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing metal-direct-source-bf16-silu-lut-route-replay-fp8-moe-block <manifest.json> <checkpoint-dir> <checkpoint-verification.json> <kernel.metal> <input.f32> <reference.f32> <output.f32> <report.json>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing metal-direct-source-bf16-reduction-width-fp8-moe-block <manifest.json> <checkpoint-dir> <checkpoint-verification.json> <kernel.metal> <input.f32> <reference.f32> <output.f32> <report.json> <lanes>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing staged-metal-fp8-expert <gate-up.safetensors> <down.safetensors> <kernel.metal> <input.f32> <reference.f32> <output.f32> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing bounded-metal-routed-row <manifest.json> <artifact-dir> <router.safetensors> <kernel.metal> <input.f32> <reference.f32> <output.f32> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing metal-fp8-expert-batch8 <gate-up.safetensors> <down.safetensors> <kernel.metal> <input.f32> <reference.f32> <output.f32>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing metal-fp8-expert-batch8-shared-weight <gate-up.safetensors> <down.safetensors> <kernel.metal> <input.f32> <reference.f32> <output.f32>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing metal-fp8-moe-block <manifest.json> <artifact-dir> <kernel.metal> <input.f32> <reference.f32> <output.f32>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing metal-noaux-tc-router <router.safetensors> <kernel.metal> <input.f32> <reference-manifest.json> <output.json>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing metal-dynamic-fp8-moe-block <manifest.json> <artifact-dir> <router.safetensors> <kernel.metal> <input.f32> <reference.f32> <output.f32>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing metal-dynamic-real-attention-fp8-moe-block <manifest.json> <artifact-dir> <router.safetensors> <kernel.metal> <source-input.f32> <candidate-input.f32> <reference.f32> <output.f32>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing metal-real-base-layer <source.safetensors> <kernel.metal> <attention-manifest.json> <hidden-input.f32> <moe-manifest.json> <artifact-dir> <router.safetensors> <source-moe-input.f32> <moe-reference.f32> <final-reference.f32> <candidate-moe-input.f32> <moe-output.f32> <final-output.f32>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing metal-union-parallel-fp8-moe-block <manifest.json> <artifact-dir> <router.safetensors> <kernel.metal> <input.f32> <reference.f32> <output.f32>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing metal-fused-gate-up-fp8-moe-block <manifest.json> <artifact-dir> <router.safetensors> <kernel.metal> <input.f32> <reference.f32> <output.f32>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing metal-simdgroup-matrix-fp8-moe-block <manifest.json> <artifact-dir> <router.safetensors> <kernel.metal> <input.f32> <reference.f32> <output.f32>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing slow-text-endpoint <checkpoint-dir> <model.lock.json> <checkpoint-verification.json> <fixture.json> <output.json> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing metal-incremental-text-endpoint <checkpoint-dir> <model.lock.json> <checkpoint-verification.json> <fixture.json> <oracle-manifest.json> <kernel.metal> <output.json> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing wide-metal-jacobi-text-endpoint <checkpoint-dir> <model.lock.json> <checkpoint-verification.json> <fixture.json> <pw0187-manifest.json> <kernel.metal> <output.json> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing arbitrary-text-generate <checkpoint-dir> <model.lock.json> <checkpoint-verification.json> <kernel.metal> <prompt.txt> <1-8-diagnostic-or-32-64-endpoint-max-tokens> <output.json> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing arbitrary-text-route-trace <checkpoint-dir> <model.lock.json> <checkpoint-verification.json> <kernel.metal> <prompt.txt> <1-8-diagnostic-or-32-64-endpoint-max-tokens> <output.json> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing arbitrary-text-first-token <checkpoint-dir> <model.lock.json> <checkpoint-verification.json> <kernel.metal> <prompt.txt> <output.json> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing weight-install-tomography <checkpoint-dir> <model.lock.json> <checkpoint-verification.json> <fixture.json> <oracle-manifest.json> <kernel.metal> <output.json> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing metal-native-distribution-probe <checkpoint-dir> <model.lock.json> <checkpoint-verification.json> <fixture.json> <oracle-manifest.json> <kernel.metal> <control|candidate> <output.json> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing layer4-metal-diagnostic <checkpoint-dir> <model.lock.json> <checkpoint-verification.json> <fixture.json> <oracle-manifest.json> <kernel.metal> <output.json> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing build-layer4-metal-ready-artifact <checkpoint-dir> <model.lock.json> <checkpoint-verification.json> <fixture.json> <oracle-manifest.json> <artifact.bin> <artifact-manifest.json> <output.json> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing benchmark-layer4-metal-ready-artifact <checkpoint-dir> <model.lock.json> <checkpoint-verification.json> <fixture.json> <oracle-manifest.json> <artifact.bin> <artifact-manifest.json> <kernel.metal> <output.json> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing benchmark-layer4-two-barrier-transaction <checkpoint-dir> <model.lock.json> <checkpoint-verification.json> <fixture.json> <oracle-manifest.json> <artifact.bin> <artifact-manifest.json> <kernel.metal> <output.json> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing benchmark-layer4-metal-native-transaction <checkpoint-dir> <model.lock.json> <checkpoint-verification.json> <fixture.json> <oracle-manifest.json> <artifact.bin> <artifact-manifest.json> <kernel.metal> <output.json> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing benchmark-metal-io-acquisition <artifact.bin> <artifact-manifest.json> <output.json> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing benchmark-pread-expert-acquisition <artifact.bin> <artifact-manifest.json> <output.json> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!("  prismwing pressure-residency-smoke <fixture.json> <output.json> <commit>");
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing pressure-resident-checkpoint-pilot <checkpoint-dir> <model.lock.json> <checkpoint-verification.json> <fixture.json> <residency-manifest.json> <resident-identity> <output.json> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing real-layer0-trace <checkpoint-dir> <model.lock.json> <checkpoint-verification.json> <fixture.json> <output-dir> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing real-layer1-routing-trace <checkpoint-dir> <model.lock.json> <checkpoint-verification.json> <fixture.json> <output-dir> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing real-layer1-expert-trace <checkpoint-dir> <model.lock.json> <checkpoint-verification.json> <fixture.json> <output-dir> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing full-prefix-trace <checkpoint-dir> <model.lock.json> <checkpoint-verification.json> <fixture.json> <output-dir> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing real-layer2-trace <checkpoint-dir> <model.lock.json> <checkpoint-verification.json> <fixture.json> <output-dir> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing real-layer4-trace <checkpoint-dir> <model.lock.json> <checkpoint-verification.json> <fixture.json> <output-dir> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing real-layer7-trace <checkpoint-dir> <model.lock.json> <checkpoint-verification.json> <fixture.json> <output-dir> <commit>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing real-routed-layer-trace <layer> <checkpoint-dir> <model.lock.json> <checkpoint-verification.json> <fixture.json> <output-dir> <commit>"
    );
    std::process::exit(2);
}

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    let result: Result<Option<PathBuf>, String> = match arguments.get(1).map(String::as_str) {
        #[cfg(target_os = "macos")]
        Some("pressure-residency-smoke") if arguments.len() == 5 => {
            let fixture = PathBuf::from(&arguments[2]);
            let output = PathBuf::from(&arguments[3]);
            run_pressure_residency_smoke(&fixture, &output, &arguments[4]).and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(Some(output))
            })
        }
        #[cfg(target_os = "macos")]
        Some("pressure-resident-checkpoint-pilot") if arguments.len() == 10 => {
            let checkpoint = PathBuf::from(&arguments[2]);
            let model_lock = PathBuf::from(&arguments[3]);
            let verification = PathBuf::from(&arguments[4]);
            let fixture = PathBuf::from(&arguments[5]);
            let residency = PathBuf::from(&arguments[6]);
            let output = PathBuf::from(&arguments[8]);
            run_pressure_resident_checkpoint_pilot(
                &checkpoint,
                &model_lock,
                &verification,
                &fixture,
                &residency,
                &arguments[7],
                &output,
                &arguments[9],
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(Some(output))
            })
        }
        #[cfg(target_os = "macos")]
        Some("metal-checkpoint-offset-probe") if arguments.len() == 5 => {
            let source = PathBuf::from(&arguments[2]);
            let output = PathBuf::from(&arguments[4]);
            run_metal_checkpoint_offset_probe(&source, &arguments[3], &output).and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(Some(output))
            })
        }
        Some("tokenize") if arguments.len() == 4 => {
            let tokenizer = Tokenizer::from_file(&arguments[2])
                .map_err(|error| format!("tokenizer load: {error}"));
            tokenizer.and_then(|tokenizer| {
                let encoded = tokenizer
                    .encode(arguments[3].clone(), false)
                    .map_err(|error| format!("tokenizer encode: {error}"))?;
                serde_json::to_writer(std::io::stdout(), encoded.get_ids())
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(None)
            })
        }
        #[cfg(target_os = "macos")]
        Some("slow-text-endpoint") if arguments.len() == 8 => {
            let checkpoint = PathBuf::from(&arguments[2]);
            let model_lock = PathBuf::from(&arguments[3]);
            let verification = PathBuf::from(&arguments[4]);
            let fixture = PathBuf::from(&arguments[5]);
            let output = PathBuf::from(&arguments[6]);
            run_slow_text_endpoint(
                &checkpoint,
                &model_lock,
                &verification,
                &fixture,
                &output,
                &arguments[7],
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(Some(output))
            })
        }
        #[cfg(target_os = "macos")]
        Some("metal-incremental-text-endpoint") if arguments.len() == 10 => {
            let checkpoint = PathBuf::from(&arguments[2]);
            let model_lock = PathBuf::from(&arguments[3]);
            let verification = PathBuf::from(&arguments[4]);
            let fixture = PathBuf::from(&arguments[5]);
            let oracle = PathBuf::from(&arguments[6]);
            let kernel = PathBuf::from(&arguments[7]);
            let output = PathBuf::from(&arguments[8]);
            run_metal_incremental_text_endpoint(
                &checkpoint,
                &model_lock,
                &verification,
                &fixture,
                &oracle,
                &kernel,
                &output,
                &arguments[9],
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(Some(output))
            })
        }
        #[cfg(target_os = "macos")]
        Some("wide-metal-jacobi-text-endpoint") if arguments.len() == 10 => {
            let checkpoint = PathBuf::from(&arguments[2]);
            let model_lock = PathBuf::from(&arguments[3]);
            let verification = PathBuf::from(&arguments[4]);
            let fixture = PathBuf::from(&arguments[5]);
            let authority = PathBuf::from(&arguments[6]);
            let kernel = PathBuf::from(&arguments[7]);
            let output = PathBuf::from(&arguments[8]);
            run_wide_metal_jacobi_text_endpoint(
                &checkpoint,
                &model_lock,
                &verification,
                &fixture,
                &authority,
                &kernel,
                &output,
                &arguments[9],
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(Some(output))
            })
        }
        #[cfg(target_os = "macos")]
        Some("arbitrary-text-generate") if arguments.len() == 10 => {
            let output_tokens = arguments[7]
                .parse::<usize>()
                .map_err(|error| format!("output token count: {error}"));
            output_tokens.and_then(|output_tokens| {
                let checkpoint = PathBuf::from(&arguments[2]);
                let model_lock = PathBuf::from(&arguments[3]);
                let verification = PathBuf::from(&arguments[4]);
                let kernel = PathBuf::from(&arguments[5]);
                let prompt = PathBuf::from(&arguments[6]);
                let output = PathBuf::from(&arguments[8]);
                run_arbitrary_text_generation(
                    &checkpoint,
                    &model_lock,
                    &verification,
                    &kernel,
                    &prompt,
                    output_tokens,
                    &output,
                    &arguments[9],
                )
                .map(|report| {
                    println!(
                        "generated {} committed tokens in {:.3} s: {}",
                        report.accepted_tokens,
                        report.complete_wall_ms / 1000.0,
                        report.generated_text
                    );
                    Some(output)
                })
            })
        }
        #[cfg(target_os = "macos")]
        Some("arbitrary-text-route-trace") if arguments.len() == 10 => {
            let output_tokens = arguments[7]
                .parse::<usize>()
                .map_err(|error| format!("output token count: {error}"));
            output_tokens.and_then(|output_tokens| {
                let checkpoint = PathBuf::from(&arguments[2]);
                let model_lock = PathBuf::from(&arguments[3]);
                let verification = PathBuf::from(&arguments[4]);
                let kernel = PathBuf::from(&arguments[5]);
                let prompt = PathBuf::from(&arguments[6]);
                let output = PathBuf::from(&arguments[8]);
                run_arbitrary_text_route_trace(
                    &checkpoint,
                    &model_lock,
                    &verification,
                    &kernel,
                    &prompt,
                    output_tokens,
                    &output,
                    &arguments[9],
                )
                .map(|report| {
                    println!(
                        "captured {} committed tokens and exact routes in {:.3} s",
                        report.accepted_tokens,
                        report.complete_wall_ms / 1000.0,
                    );
                    Some(output)
                })
            })
        }
        #[cfg(target_os = "macos")]
        Some("arbitrary-text-first-token") if arguments.len() == 9 => {
            let checkpoint = PathBuf::from(&arguments[2]);
            let model_lock = PathBuf::from(&arguments[3]);
            let verification = PathBuf::from(&arguments[4]);
            let kernel = PathBuf::from(&arguments[5]);
            let prompt = PathBuf::from(&arguments[6]);
            let output = PathBuf::from(&arguments[7]);
            run_arbitrary_text_generation(
                &checkpoint,
                &model_lock,
                &verification,
                &kernel,
                &prompt,
                1,
                &output,
                &arguments[8],
            )
            .map(|report| {
                println!(
                    "first token {} in {:.3} s: {}",
                    report.generated_token_ids[0],
                    report.complete_wall_ms / 1000.0,
                    report.generated_text
                );
                Some(output)
            })
        }
        #[cfg(target_os = "macos")]
        Some("metal-native-distribution-probe") if arguments.len() == 11 => {
            let checkpoint = PathBuf::from(&arguments[2]);
            let model_lock = PathBuf::from(&arguments[3]);
            let verification = PathBuf::from(&arguments[4]);
            let fixture = PathBuf::from(&arguments[5]);
            let oracle = PathBuf::from(&arguments[6]);
            let kernel = PathBuf::from(&arguments[7]);
            let repair_mode = &arguments[8];
            let output = PathBuf::from(&arguments[9]);
            run_metal_native_distribution_probe(
                &checkpoint,
                &model_lock,
                &verification,
                &fixture,
                &oracle,
                &kernel,
                repair_mode,
                &output,
                &arguments[10],
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(Some(output))
            })
        }
        #[cfg(target_os = "macos")]
        Some("weight-install-tomography") if arguments.len() == 10 => {
            let checkpoint = PathBuf::from(&arguments[2]);
            let model_lock = PathBuf::from(&arguments[3]);
            let verification = PathBuf::from(&arguments[4]);
            let fixture = PathBuf::from(&arguments[5]);
            let oracle = PathBuf::from(&arguments[6]);
            let kernel = PathBuf::from(&arguments[7]);
            let output = PathBuf::from(&arguments[8]);
            run_weight_install_tomography(
                &checkpoint,
                &model_lock,
                &verification,
                &fixture,
                &oracle,
                &kernel,
                &output,
                &arguments[9],
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(Some(output))
            })
        }
        #[cfg(target_os = "macos")]
        Some("layer4-metal-diagnostic") if arguments.len() == 10 => {
            let checkpoint = PathBuf::from(&arguments[2]);
            let model_lock = PathBuf::from(&arguments[3]);
            let verification = PathBuf::from(&arguments[4]);
            let fixture = PathBuf::from(&arguments[5]);
            let oracle = PathBuf::from(&arguments[6]);
            let kernel = PathBuf::from(&arguments[7]);
            let output = PathBuf::from(&arguments[8]);
            run_layer4_metal_diagnostic(
                &checkpoint,
                &model_lock,
                &verification,
                &fixture,
                &oracle,
                &kernel,
                &output,
                &arguments[9],
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(Some(output))
            })
        }
        #[cfg(target_os = "macos")]
        Some("build-layer4-metal-ready-artifact") if arguments.len() == 11 => {
            let checkpoint = PathBuf::from(&arguments[2]);
            let model_lock = PathBuf::from(&arguments[3]);
            let verification = PathBuf::from(&arguments[4]);
            let fixture = PathBuf::from(&arguments[5]);
            let oracle = PathBuf::from(&arguments[6]);
            let artifact = PathBuf::from(&arguments[7]);
            let manifest = PathBuf::from(&arguments[8]);
            let output = PathBuf::from(&arguments[9]);
            build_layer4_metal_ready_artifact(
                &checkpoint,
                &model_lock,
                &verification,
                &fixture,
                &oracle,
                &artifact,
                &manifest,
                &output,
                &arguments[10],
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(Some(output))
            })
        }
        #[cfg(target_os = "macos")]
        Some("benchmark-layer4-metal-ready-artifact") if arguments.len() == 12 => {
            let checkpoint = PathBuf::from(&arguments[2]);
            let model_lock = PathBuf::from(&arguments[3]);
            let verification = PathBuf::from(&arguments[4]);
            let fixture = PathBuf::from(&arguments[5]);
            let oracle = PathBuf::from(&arguments[6]);
            let artifact = PathBuf::from(&arguments[7]);
            let manifest = PathBuf::from(&arguments[8]);
            let kernel = PathBuf::from(&arguments[9]);
            let output = PathBuf::from(&arguments[10]);
            benchmark_layer4_metal_ready_artifact(
                &checkpoint,
                &model_lock,
                &verification,
                &fixture,
                &oracle,
                &artifact,
                &manifest,
                &kernel,
                &output,
                &arguments[11],
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(Some(output))
            })
        }
        #[cfg(target_os = "macos")]
        Some("benchmark-layer4-two-barrier-transaction") if arguments.len() == 12 => {
            let checkpoint = PathBuf::from(&arguments[2]);
            let model_lock = PathBuf::from(&arguments[3]);
            let verification = PathBuf::from(&arguments[4]);
            let fixture = PathBuf::from(&arguments[5]);
            let oracle = PathBuf::from(&arguments[6]);
            let artifact = PathBuf::from(&arguments[7]);
            let manifest = PathBuf::from(&arguments[8]);
            let kernel = PathBuf::from(&arguments[9]);
            let output = PathBuf::from(&arguments[10]);
            benchmark_layer4_two_barrier_transaction(
                &checkpoint,
                &model_lock,
                &verification,
                &fixture,
                &oracle,
                &artifact,
                &manifest,
                &kernel,
                &output,
                &arguments[11],
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(Some(output))
            })
        }
        #[cfg(target_os = "macos")]
        Some("benchmark-layer4-metal-native-transaction") if arguments.len() == 12 => {
            let checkpoint = PathBuf::from(&arguments[2]);
            let model_lock = PathBuf::from(&arguments[3]);
            let verification = PathBuf::from(&arguments[4]);
            let fixture = PathBuf::from(&arguments[5]);
            let oracle = PathBuf::from(&arguments[6]);
            let artifact = PathBuf::from(&arguments[7]);
            let manifest = PathBuf::from(&arguments[8]);
            let kernel = PathBuf::from(&arguments[9]);
            let output = PathBuf::from(&arguments[10]);
            benchmark_layer4_metal_native_transaction(
                &checkpoint,
                &model_lock,
                &verification,
                &fixture,
                &oracle,
                &artifact,
                &manifest,
                &kernel,
                &output,
                &arguments[11],
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(Some(output))
            })
        }
        #[cfg(target_os = "macos")]
        Some("benchmark-metal-io-acquisition") if arguments.len() == 6 => {
            let artifact = PathBuf::from(&arguments[2]);
            let manifest = PathBuf::from(&arguments[3]);
            let output = PathBuf::from(&arguments[4]);
            benchmark_metal_io_acquisition(&artifact, &manifest, &output, &arguments[5]).and_then(
                |report| {
                    serde_json::to_writer(std::io::stdout(), &report)
                        .map_err(|error| error.to_string())?;
                    println!();
                    Ok(Some(output))
                },
            )
        }
        #[cfg(target_os = "macos")]
        Some("benchmark-pread-expert-acquisition") if arguments.len() == 6 => {
            let artifact = PathBuf::from(&arguments[2]);
            let manifest = PathBuf::from(&arguments[3]);
            let output = PathBuf::from(&arguments[4]);
            benchmark_pread_expert_acquisition(&artifact, &manifest, &output, &arguments[5])
                .and_then(|report| {
                    serde_json::to_writer(std::io::stdout(), &report)
                        .map_err(|error| error.to_string())?;
                    println!();
                    Ok(Some(output))
                })
        }
        #[cfg(target_os = "macos")]
        Some("real-layer0-trace") if arguments.len() == 8 => {
            let checkpoint = PathBuf::from(&arguments[2]);
            let model_lock = PathBuf::from(&arguments[3]);
            let verification = PathBuf::from(&arguments[4]);
            let fixture = PathBuf::from(&arguments[5]);
            let output = PathBuf::from(&arguments[6]);
            run_real_layer0_trace(
                &checkpoint,
                &model_lock,
                &verification,
                &fixture,
                &output,
                &arguments[7],
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(Some(output.join("manifest.json")))
            })
        }
        #[cfg(target_os = "macos")]
        Some("real-layer1-routing-trace") if arguments.len() == 8 => {
            let checkpoint = PathBuf::from(&arguments[2]);
            let model_lock = PathBuf::from(&arguments[3]);
            let verification = PathBuf::from(&arguments[4]);
            let fixture = PathBuf::from(&arguments[5]);
            let output = PathBuf::from(&arguments[6]);
            run_real_layer1_routing_trace(
                &checkpoint,
                &model_lock,
                &verification,
                &fixture,
                &output,
                &arguments[7],
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(Some(output.join("manifest.json")))
            })
        }
        #[cfg(target_os = "macos")]
        Some("real-layer1-expert-trace") if arguments.len() == 8 => {
            let checkpoint = PathBuf::from(&arguments[2]);
            let model_lock = PathBuf::from(&arguments[3]);
            let verification = PathBuf::from(&arguments[4]);
            let fixture = PathBuf::from(&arguments[5]);
            let output = PathBuf::from(&arguments[6]);
            run_real_layer1_expert_trace(
                &checkpoint,
                &model_lock,
                &verification,
                &fixture,
                &output,
                &arguments[7],
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(Some(output.join("manifest.json")))
            })
        }
        #[cfg(target_os = "macos")]
        Some("routed-mixture-activation-corpus") if arguments.len() == 9 => {
            let checkpoint = PathBuf::from(&arguments[2]);
            let model_lock = PathBuf::from(&arguments[3]);
            let verification = PathBuf::from(&arguments[4]);
            let fixture = PathBuf::from(&arguments[5]);
            let pw0112_manifest = PathBuf::from(&arguments[6]);
            let output = PathBuf::from(&arguments[7]);
            run_routed_mixture_activation_corpus(
                &checkpoint,
                &model_lock,
                &verification,
                &fixture,
                &pw0112_manifest,
                &output,
                &arguments[8],
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(Some(output.join("manifest.json")))
            })
        }
        #[cfg(target_os = "macos")]
        Some("route-only-trace") if arguments.len() == 8 => {
            let checkpoint = PathBuf::from(&arguments[2]);
            let model_lock = PathBuf::from(&arguments[3]);
            let verification = PathBuf::from(&arguments[4]);
            let fixture = PathBuf::from(&arguments[5]);
            let output = PathBuf::from(&arguments[6]);
            run_route_only_trace(
                &checkpoint,
                &model_lock,
                &verification,
                &fixture,
                &output,
                &arguments[7],
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(Some(output.join("manifest.json")))
            })
        }
        #[cfg(target_os = "macos")]
        Some("prefill-route-coverage-trace") if arguments.len() == 9 => {
            let checkpoint = PathBuf::from(&arguments[2]);
            let model_lock = PathBuf::from(&arguments[3]);
            let verification = PathBuf::from(&arguments[4]);
            let fixture = PathBuf::from(&arguments[5]);
            let positions = arguments[6]
                .parse::<usize>()
                .map_err(|error| format!("invalid prefix positions: {error}"));
            let output = PathBuf::from(&arguments[7]);
            positions.and_then(|positions| {
                run_prefill_route_coverage_trace(
                    &checkpoint,
                    &model_lock,
                    &verification,
                    &fixture,
                    positions,
                    &output,
                    &arguments[8],
                )
                .and_then(|report| {
                    serde_json::to_writer(std::io::stdout(), &report)
                        .map_err(|error| error.to_string())?;
                    println!();
                    Ok(Some(output.join("manifest.json")))
                })
            })
        }
        #[cfg(target_os = "macos")]
        Some("global-attention-sparsity-trace") if arguments.len() == 9 => {
            let checkpoint = PathBuf::from(&arguments[2]);
            let model_lock = PathBuf::from(&arguments[3]);
            let verification = PathBuf::from(&arguments[4]);
            let fixture = PathBuf::from(&arguments[5]);
            let pw0157_prefix512 = PathBuf::from(&arguments[6]);
            let output = PathBuf::from(&arguments[7]);
            run_global_attention_sparsity_trace(
                &checkpoint,
                &model_lock,
                &verification,
                &fixture,
                &pw0157_prefix512,
                &output,
                &arguments[8],
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(Some(output.join("manifest.json")))
            })
        }
        #[cfg(target_os = "macos")]
        Some("structured-sparse-layer0-trace") if arguments.len() == 9 => {
            let checkpoint = PathBuf::from(&arguments[2]);
            let model_lock = PathBuf::from(&arguments[3]);
            let verification = PathBuf::from(&arguments[4]);
            let authority_fixture = PathBuf::from(&arguments[5]);
            let pw0176_fixture = PathBuf::from(&arguments[6]);
            let output = PathBuf::from(&arguments[7]);
            run_structured_sparse_layer0_trace(
                &checkpoint,
                &model_lock,
                &verification,
                &authority_fixture,
                &pw0176_fixture,
                &output,
                &arguments[8],
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(Some(output.join("manifest.json")))
            })
        }
        #[cfg(target_os = "macos")]
        Some("global-attention-capture-smoke") if arguments.len() == 9 => {
            let checkpoint = PathBuf::from(&arguments[2]);
            let model_lock = PathBuf::from(&arguments[3]);
            let verification = PathBuf::from(&arguments[4]);
            let fixture = PathBuf::from(&arguments[5]);
            let route_authority = PathBuf::from(&arguments[6]);
            let output = PathBuf::from(&arguments[7]);
            run_global_attention_capture_smoke(
                &checkpoint,
                &model_lock,
                &verification,
                &fixture,
                &route_authority,
                &output,
                &arguments[8],
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(Some(output.join("manifest.json")))
            })
        }
        #[cfg(target_os = "macos")]
        Some("full-prefix-trace") if arguments.len() == 8 => {
            let checkpoint = PathBuf::from(&arguments[2]);
            let model_lock = PathBuf::from(&arguments[3]);
            let verification = PathBuf::from(&arguments[4]);
            let fixture = PathBuf::from(&arguments[5]);
            let output = PathBuf::from(&arguments[6]);
            run_full_prefix_trace(
                &checkpoint,
                &model_lock,
                &verification,
                &fixture,
                &output,
                &arguments[7],
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(Some(output.join("manifest.json")))
            })
        }
        #[cfg(target_os = "macos")]
        Some("real-layer2-trace") if arguments.len() == 8 => {
            let checkpoint = PathBuf::from(&arguments[2]);
            let model_lock = PathBuf::from(&arguments[3]);
            let verification = PathBuf::from(&arguments[4]);
            let fixture = PathBuf::from(&arguments[5]);
            let output = PathBuf::from(&arguments[6]);
            run_real_layer2_trace(
                &checkpoint,
                &model_lock,
                &verification,
                &fixture,
                &output,
                &arguments[7],
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(Some(output.join("manifest.json")))
            })
        }
        #[cfg(target_os = "macos")]
        Some("real-layer4-trace") if arguments.len() == 8 => {
            let checkpoint = PathBuf::from(&arguments[2]);
            let model_lock = PathBuf::from(&arguments[3]);
            let verification = PathBuf::from(&arguments[4]);
            let fixture = PathBuf::from(&arguments[5]);
            let output = PathBuf::from(&arguments[6]);
            run_real_layer4_trace(
                &checkpoint,
                &model_lock,
                &verification,
                &fixture,
                &output,
                &arguments[7],
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(Some(output.join("manifest.json")))
            })
        }
        #[cfg(target_os = "macos")]
        Some("real-layer7-trace") if arguments.len() == 8 => {
            let checkpoint = PathBuf::from(&arguments[2]);
            let model_lock = PathBuf::from(&arguments[3]);
            let verification = PathBuf::from(&arguments[4]);
            let fixture = PathBuf::from(&arguments[5]);
            let output = PathBuf::from(&arguments[6]);
            run_real_layer7_trace(
                &checkpoint,
                &model_lock,
                &verification,
                &fixture,
                &output,
                &arguments[7],
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(Some(output.join("manifest.json")))
            })
        }
        #[cfg(target_os = "macos")]
        Some("real-routed-layer-trace") if arguments.len() == 9 => {
            let target_layer = arguments[2]
                .parse::<usize>()
                .map_err(|error| format!("invalid routed trace layer: {error}"));
            target_layer.and_then(|target_layer| {
                let checkpoint = PathBuf::from(&arguments[3]);
                let model_lock = PathBuf::from(&arguments[4]);
                let verification = PathBuf::from(&arguments[5]);
                let fixture = PathBuf::from(&arguments[6]);
                let output = PathBuf::from(&arguments[7]);
                run_real_routed_layer_trace(
                    &checkpoint,
                    &model_lock,
                    &verification,
                    &fixture,
                    &output,
                    &arguments[8],
                    target_layer,
                )
                .and_then(|report| {
                    serde_json::to_writer(std::io::stdout(), &report)
                        .map_err(|error| error.to_string())?;
                    println!();
                    Ok(Some(output.join("manifest.json")))
                })
            })
        }
        Some("census") if arguments.len() == 5 => {
            let index = PathBuf::from(&arguments[2]);
            let checkpoint = PathBuf::from(&arguments[3]);
            let output = PathBuf::from(&arguments[4]);
            build_census(&index, &checkpoint)
                .and_then(|census| write_census(&census, &output))
                .map(|_| Some(output))
        }
        Some("repack") if arguments.len() >= 5 => {
            let source = PathBuf::from(&arguments[2]);
            let output = PathBuf::from(&arguments[3]);
            repack_expert_container(&source, &output, &arguments[4..]).map(|_| Some(output))
        }
        Some("verify-container") if arguments.len() == 3 => {
            let container = PathBuf::from(&arguments[2]);
            verify_expert_container(&container).map(|_| Some(container))
        }
        Some("inspect-tensor") if arguments.len() == 4 => {
            let source = PathBuf::from(&arguments[2]);
            inspect_mapped_tensor(&source, &arguments[3]).and_then(|inspection| {
                serde_json::to_writer(std::io::stdout(), &inspection)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(None)
            })
        }
        Some("fp8-gemv") if arguments.len() == 7 => {
            let source = PathBuf::from(&arguments[2]);
            let input = PathBuf::from(&arguments[5]);
            let output = PathBuf::from(&arguments[6]);
            run_mapped_fp8_gemv(&source, &arguments[3], &arguments[4], &input, &output).and_then(
                |report| {
                    serde_json::to_writer(std::io::stdout(), &report)
                        .map_err(|error| error.to_string())?;
                    println!();
                    Ok(None)
                },
            )
        }
        #[cfg(target_os = "macos")]
        Some("metal-fp8-gemv") if arguments.len() == 9 => {
            let source = PathBuf::from(&arguments[2]);
            let kernel = PathBuf::from(&arguments[3]);
            let input = PathBuf::from(&arguments[6]);
            let reference = PathBuf::from(&arguments[7]);
            let output = PathBuf::from(&arguments[8]);
            run_metal_mapped_fp8_gemv(
                &source,
                &kernel,
                &arguments[4],
                &arguments[5],
                &input,
                &reference,
                &output,
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(None)
            })
        }
        #[cfg(target_os = "macos")]
        Some("metal-direct-fp8-gemv") if arguments.len() == 10 => {
            let source = PathBuf::from(&arguments[2]);
            let kernel = PathBuf::from(&arguments[3]);
            let input = PathBuf::from(&arguments[6]);
            let reference = PathBuf::from(&arguments[7]);
            let output = PathBuf::from(&arguments[8]);
            let report_path = PathBuf::from(&arguments[9]);
            if report_path.exists() {
                Err(format!("refusing to overwrite {}", report_path.display()))
            } else {
                run_metal_direct_mapped_fp8_gemv(
                    &source,
                    &kernel,
                    &arguments[4],
                    &arguments[5],
                    &input,
                    &reference,
                    &output,
                )
                .and_then(|report| {
                    let report_file = OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&report_path)
                        .map_err(|error| format!("{}: {error}", report_path.display()))?;
                    serde_json::to_writer_pretty(report_file, &report)
                        .map_err(|error| error.to_string())?;
                    serde_json::to_writer(std::io::stdout(), &report)
                        .map_err(|error| error.to_string())?;
                    println!();
                    Ok(None)
                })
            }
        }
        #[cfg(target_os = "macos")]
        Some("metal-direct-source-bf16-fp8-gemv") if arguments.len() == 10 => {
            let source = PathBuf::from(&arguments[2]);
            let kernel = PathBuf::from(&arguments[3]);
            let input = PathBuf::from(&arguments[6]);
            let reference = PathBuf::from(&arguments[7]);
            let output = PathBuf::from(&arguments[8]);
            let report_path = PathBuf::from(&arguments[9]);
            if report_path.exists() {
                Err(format!("refusing to overwrite {}", report_path.display()))
            } else {
                run_metal_direct_source_bf16_fp8_gemv(
                    &source,
                    &kernel,
                    &arguments[4],
                    &arguments[5],
                    &input,
                    &reference,
                    &output,
                )
                .and_then(|report| {
                    let report_file = OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&report_path)
                        .map_err(|error| format!("{}: {error}", report_path.display()))?;
                    serde_json::to_writer_pretty(report_file, &report)
                        .map_err(|error| error.to_string())?;
                    serde_json::to_writer(std::io::stdout(), &report)
                        .map_err(|error| error.to_string())?;
                    println!();
                    Ok(None)
                })
            }
        }
        #[cfg(target_os = "macos")]
        Some("metal-direct-source-bf16-fp8-gemv-audit") if arguments.len() == 10 => {
            let source = PathBuf::from(&arguments[2]);
            let kernel = PathBuf::from(&arguments[3]);
            let input = PathBuf::from(&arguments[6]);
            let reference = PathBuf::from(&arguments[7]);
            let output = PathBuf::from(&arguments[8]);
            let report_path = PathBuf::from(&arguments[9]);
            if report_path.exists() {
                Err(format!("refusing to overwrite {}", report_path.display()))
            } else {
                run_metal_direct_source_bf16_fp8_gemv_audit(
                    &source,
                    &kernel,
                    &arguments[4],
                    &arguments[5],
                    &input,
                    &reference,
                    &output,
                )
                .and_then(|report| {
                    let report_file = OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&report_path)
                        .map_err(|error| format!("{}: {error}", report_path.display()))?;
                    serde_json::to_writer_pretty(report_file, &report)
                        .map_err(|error| error.to_string())?;
                    serde_json::to_writer(std::io::stdout(), &report)
                        .map_err(|error| error.to_string())?;
                    println!();
                    Ok(None)
                })
            }
        }
        #[cfg(target_os = "macos")]
        Some("metal-fp8-expert") if arguments.len() == 8 => {
            let gate_up = PathBuf::from(&arguments[2]);
            let down = PathBuf::from(&arguments[3]);
            let kernel = PathBuf::from(&arguments[4]);
            let input = PathBuf::from(&arguments[5]);
            let reference = PathBuf::from(&arguments[6]);
            let output = PathBuf::from(&arguments[7]);
            run_metal_fp8_expert(&gate_up, &down, &kernel, &input, &reference, &output).and_then(
                |report| {
                    serde_json::to_writer(std::io::stdout(), &report)
                        .map_err(|error| error.to_string())?;
                    println!();
                    Ok(None)
                },
            )
        }
        #[cfg(target_os = "macos")]
        Some("metal-direct-fp8-expert") if arguments.len() == 9 => {
            let gate_up = PathBuf::from(&arguments[2]);
            let down = PathBuf::from(&arguments[3]);
            let kernel = PathBuf::from(&arguments[4]);
            let input = PathBuf::from(&arguments[5]);
            let reference = PathBuf::from(&arguments[6]);
            let output = PathBuf::from(&arguments[7]);
            let report_path = PathBuf::from(&arguments[8]);
            if report_path.exists() {
                Err(format!("refusing to overwrite {}", report_path.display()))
            } else {
                run_metal_direct_fp8_expert(&gate_up, &down, &kernel, &input, &reference, &output)
                    .and_then(|report| {
                        let report_file = OpenOptions::new()
                            .write(true)
                            .create_new(true)
                            .open(&report_path)
                            .map_err(|error| format!("{}: {error}", report_path.display()))?;
                        serde_json::to_writer_pretty(report_file, &report)
                            .map_err(|error| error.to_string())?;
                        serde_json::to_writer(std::io::stdout(), &report)
                            .map_err(|error| error.to_string())?;
                        println!();
                        Ok(None)
                    })
            }
        }
        #[cfg(target_os = "macos")]
        Some("metal-direct-fp8-expert-batch8-shared-weight") if arguments.len() == 9 => {
            let gate_up = PathBuf::from(&arguments[2]);
            let down = PathBuf::from(&arguments[3]);
            let kernel = PathBuf::from(&arguments[4]);
            let input = PathBuf::from(&arguments[5]);
            let reference = PathBuf::from(&arguments[6]);
            let output = PathBuf::from(&arguments[7]);
            let report_path = PathBuf::from(&arguments[8]);
            if report_path.exists() {
                Err(format!("refusing to overwrite {}", report_path.display()))
            } else {
                run_metal_direct_fp8_expert_batch8_shared_weight(
                    &gate_up, &down, &kernel, &input, &reference, &output,
                )
                .and_then(|report| {
                    let report_file = OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&report_path)
                        .map_err(|error| format!("{}: {error}", report_path.display()))?;
                    serde_json::to_writer_pretty(report_file, &report)
                        .map_err(|error| error.to_string())?;
                    serde_json::to_writer(std::io::stdout(), &report)
                        .map_err(|error| error.to_string())?;
                    println!();
                    Ok(None)
                })
            }
        }
        #[cfg(target_os = "macos")]
        Some("metal-direct-route-replay-fp8-moe-block") if arguments.len() == 10 => {
            let manifest = PathBuf::from(&arguments[2]);
            let checkpoint = PathBuf::from(&arguments[3]);
            let verification = PathBuf::from(&arguments[4]);
            let kernel = PathBuf::from(&arguments[5]);
            let input = PathBuf::from(&arguments[6]);
            let reference = PathBuf::from(&arguments[7]);
            let output = PathBuf::from(&arguments[8]);
            let report_path = PathBuf::from(&arguments[9]);
            if report_path.exists() {
                Err(format!("refusing to overwrite {}", report_path.display()))
            } else {
                run_metal_direct_route_replay_fp8_moe_block(
                    &manifest,
                    &checkpoint,
                    &verification,
                    &kernel,
                    &input,
                    &reference,
                    &output,
                )
                .and_then(|report| {
                    let report_file = OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&report_path)
                        .map_err(|error| format!("{}: {error}", report_path.display()))?;
                    serde_json::to_writer_pretty(report_file, &report)
                        .map_err(|error| error.to_string())?;
                    serde_json::to_writer(std::io::stdout(), &report)
                        .map_err(|error| error.to_string())?;
                    println!();
                    Ok(None)
                })
            }
        }
        #[cfg(target_os = "macos")]
        Some("metal-direct-source-bf16-route-replay-fp8-moe-block") if arguments.len() == 10 => {
            let manifest = PathBuf::from(&arguments[2]);
            let checkpoint = PathBuf::from(&arguments[3]);
            let verification = PathBuf::from(&arguments[4]);
            let kernel = PathBuf::from(&arguments[5]);
            let input = PathBuf::from(&arguments[6]);
            let reference = PathBuf::from(&arguments[7]);
            let output = PathBuf::from(&arguments[8]);
            let report_path = PathBuf::from(&arguments[9]);
            if report_path.exists() {
                Err(format!("refusing to overwrite {}", report_path.display()))
            } else {
                run_metal_direct_source_bf16_route_replay_fp8_moe_block(
                    &manifest,
                    &checkpoint,
                    &verification,
                    &kernel,
                    &input,
                    &reference,
                    &output,
                )
                .and_then(|report| {
                    let report_file = OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&report_path)
                        .map_err(|error| format!("{}: {error}", report_path.display()))?;
                    serde_json::to_writer_pretty(report_file, &report)
                        .map_err(|error| error.to_string())?;
                    serde_json::to_writer(std::io::stdout(), &report)
                        .map_err(|error| error.to_string())?;
                    println!();
                    Ok(None)
                })
            }
        }
        #[cfg(target_os = "macos")]
        Some("metal-direct-source-bf16-silu-lut-route-replay-fp8-moe-block")
            if arguments.len() == 10 =>
        {
            let manifest = PathBuf::from(&arguments[2]);
            let checkpoint = PathBuf::from(&arguments[3]);
            let verification = PathBuf::from(&arguments[4]);
            let kernel = PathBuf::from(&arguments[5]);
            let input = PathBuf::from(&arguments[6]);
            let reference = PathBuf::from(&arguments[7]);
            let output = PathBuf::from(&arguments[8]);
            let report_path = PathBuf::from(&arguments[9]);
            if report_path.exists() {
                Err(format!("refusing to overwrite {}", report_path.display()))
            } else {
                run_metal_direct_source_bf16_silu_lut_route_replay_fp8_moe_block(
                    &manifest,
                    &checkpoint,
                    &verification,
                    &kernel,
                    &input,
                    &reference,
                    &output,
                )
                .and_then(|report| {
                    let report_file = OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&report_path)
                        .map_err(|error| format!("{}: {error}", report_path.display()))?;
                    serde_json::to_writer_pretty(report_file, &report)
                        .map_err(|error| error.to_string())?;
                    serde_json::to_writer(std::io::stdout(), &report)
                        .map_err(|error| error.to_string())?;
                    println!();
                    Ok(None)
                })
            }
        }
        #[cfg(target_os = "macos")]
        Some("metal-direct-source-bf16-reduction-width-fp8-moe-block") if arguments.len() == 11 => {
            let manifest = PathBuf::from(&arguments[2]);
            let checkpoint = PathBuf::from(&arguments[3]);
            let verification = PathBuf::from(&arguments[4]);
            let kernel = PathBuf::from(&arguments[5]);
            let input = PathBuf::from(&arguments[6]);
            let reference = PathBuf::from(&arguments[7]);
            let output = PathBuf::from(&arguments[8]);
            let report_path = PathBuf::from(&arguments[9]);
            let lanes = arguments[10]
                .parse::<u64>()
                .map_err(|error| format!("invalid projection lanes: {error}"));
            if report_path.exists() {
                Err(format!("refusing to overwrite {}", report_path.display()))
            } else {
                lanes.and_then(|lanes| {
                    run_metal_direct_source_bf16_reduction_width_fp8_moe_block(
                        &manifest,
                        &checkpoint,
                        &verification,
                        &kernel,
                        &input,
                        &reference,
                        &output,
                        lanes,
                    )
                    .and_then(|report| {
                        let report_file = OpenOptions::new()
                            .write(true)
                            .create_new(true)
                            .open(&report_path)
                            .map_err(|error| format!("{}: {error}", report_path.display()))?;
                        serde_json::to_writer_pretty(report_file, &report)
                            .map_err(|error| error.to_string())?;
                        serde_json::to_writer(std::io::stdout(), &report)
                            .map_err(|error| error.to_string())?;
                        println!();
                        Ok(None)
                    })
                })
            }
        }
        #[cfg(target_os = "macos")]
        Some("staged-metal-fp8-expert") if arguments.len() == 9 => {
            let gate_up = PathBuf::from(&arguments[2]);
            let down = PathBuf::from(&arguments[3]);
            let kernel = PathBuf::from(&arguments[4]);
            let input = PathBuf::from(&arguments[5]);
            let reference = PathBuf::from(&arguments[6]);
            let output = PathBuf::from(&arguments[7]);
            run_staged_metal_fp8_expert(
                &gate_up,
                &down,
                &kernel,
                &input,
                &reference,
                &output,
                &arguments[8],
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(None)
            })
        }
        #[cfg(target_os = "macos")]
        Some("bounded-metal-routed-row") if arguments.len() == 10 => {
            let manifest = PathBuf::from(&arguments[2]);
            let artifacts = PathBuf::from(&arguments[3]);
            let router = PathBuf::from(&arguments[4]);
            let kernel = PathBuf::from(&arguments[5]);
            let input = PathBuf::from(&arguments[6]);
            let reference = PathBuf::from(&arguments[7]);
            let output = PathBuf::from(&arguments[8]);
            run_bounded_metal_routed_row(
                &manifest,
                &artifacts,
                &router,
                &kernel,
                &input,
                &reference,
                &output,
                &arguments[9],
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(None)
            })
        }
        #[cfg(target_os = "macos")]
        Some("metal-base-layer-attention") if arguments.len() == 7 => {
            let source = PathBuf::from(&arguments[2]);
            let kernel = PathBuf::from(&arguments[3]);
            let manifest = PathBuf::from(&arguments[4]);
            let input = PathBuf::from(&arguments[5]);
            let output = PathBuf::from(&arguments[6]);
            run_metal_base_layer_attention(&source, &kernel, &manifest, &input, &output).and_then(
                |report| {
                    serde_json::to_writer(std::io::stdout(), &report)
                        .map_err(|error| error.to_string())?;
                    println!();
                    Ok(None)
                },
            )
        }
        #[cfg(target_os = "macos")]
        Some("metal-fp8-expert-batch8") if arguments.len() == 8 => {
            let gate_up = PathBuf::from(&arguments[2]);
            let down = PathBuf::from(&arguments[3]);
            let kernel = PathBuf::from(&arguments[4]);
            let input = PathBuf::from(&arguments[5]);
            let reference = PathBuf::from(&arguments[6]);
            let output = PathBuf::from(&arguments[7]);
            run_metal_fp8_expert_batch8(&gate_up, &down, &kernel, &input, &reference, &output)
                .and_then(|report| {
                    serde_json::to_writer(std::io::stdout(), &report)
                        .map_err(|error| error.to_string())?;
                    println!();
                    Ok(None)
                })
        }
        #[cfg(target_os = "macos")]
        Some("metal-fp8-expert-batch8-shared-weight") if arguments.len() == 8 => {
            let gate_up = PathBuf::from(&arguments[2]);
            let down = PathBuf::from(&arguments[3]);
            let kernel = PathBuf::from(&arguments[4]);
            let input = PathBuf::from(&arguments[5]);
            let reference = PathBuf::from(&arguments[6]);
            let output = PathBuf::from(&arguments[7]);
            run_metal_fp8_expert_batch8_shared_weight(
                &gate_up, &down, &kernel, &input, &reference, &output,
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(None)
            })
        }
        #[cfg(target_os = "macos")]
        Some("metal-fp8-moe-block") if arguments.len() == 8 => {
            let manifest = PathBuf::from(&arguments[2]);
            let artifacts = PathBuf::from(&arguments[3]);
            let kernel = PathBuf::from(&arguments[4]);
            let input = PathBuf::from(&arguments[5]);
            let reference = PathBuf::from(&arguments[6]);
            let output = PathBuf::from(&arguments[7]);
            run_metal_fp8_moe_block(&manifest, &artifacts, &kernel, &input, &reference, &output)
                .and_then(|report| {
                    serde_json::to_writer(std::io::stdout(), &report)
                        .map_err(|error| error.to_string())?;
                    println!();
                    Ok(None)
                })
        }
        #[cfg(target_os = "macos")]
        Some("metal-noaux-tc-router") if arguments.len() == 7 => {
            let router = PathBuf::from(&arguments[2]);
            let kernel = PathBuf::from(&arguments[3]);
            let input = PathBuf::from(&arguments[4]);
            let reference = PathBuf::from(&arguments[5]);
            let output = PathBuf::from(&arguments[6]);
            run_metal_noaux_tc_router(&router, &kernel, &input, &reference, &output).and_then(
                |report| {
                    serde_json::to_writer(std::io::stdout(), &report)
                        .map_err(|error| error.to_string())?;
                    println!();
                    Ok(None)
                },
            )
        }
        #[cfg(target_os = "macos")]
        Some("metal-dynamic-fp8-moe-block") if arguments.len() == 9 => {
            let manifest = PathBuf::from(&arguments[2]);
            let artifacts = PathBuf::from(&arguments[3]);
            let router = PathBuf::from(&arguments[4]);
            let kernel = PathBuf::from(&arguments[5]);
            let input = PathBuf::from(&arguments[6]);
            let reference = PathBuf::from(&arguments[7]);
            let output = PathBuf::from(&arguments[8]);
            run_metal_dynamic_fp8_moe_block(
                &manifest, &artifacts, &router, &kernel, &input, &reference, &output,
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(None)
            })
        }
        #[cfg(target_os = "macos")]
        Some("metal-dynamic-real-attention-fp8-moe-block") if arguments.len() == 10 => {
            let manifest = PathBuf::from(&arguments[2]);
            let artifacts = PathBuf::from(&arguments[3]);
            let router = PathBuf::from(&arguments[4]);
            let kernel = PathBuf::from(&arguments[5]);
            let source_input = PathBuf::from(&arguments[6]);
            let candidate_input = PathBuf::from(&arguments[7]);
            let reference = PathBuf::from(&arguments[8]);
            let output = PathBuf::from(&arguments[9]);
            run_metal_dynamic_real_attention_fp8_moe_block(RealAttentionMoeRequest {
                manifest_path: &manifest,
                artifact_root: &artifacts,
                router_path: &router,
                kernel_path: &kernel,
                source_input_path: &source_input,
                candidate_input_path: &candidate_input,
                reference_path: &reference,
                output_path: &output,
            })
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(None)
            })
        }
        #[cfg(target_os = "macos")]
        Some("metal-real-base-layer") if arguments.len() == 15 => {
            let paths = arguments[2..].iter().map(PathBuf::from).collect::<Vec<_>>();
            run_metal_real_base_layer(RealBaseLayerRequest {
                source_path: &paths[0],
                kernel_path: &paths[1],
                attention_manifest_path: &paths[2],
                hidden_input_path: &paths[3],
                moe_manifest_path: &paths[4],
                artifact_root: &paths[5],
                router_path: &paths[6],
                source_moe_input_path: &paths[7],
                moe_reference_path: &paths[8],
                final_reference_path: &paths[9],
                candidate_moe_input_path: &paths[10],
                moe_output_path: &paths[11],
                final_output_path: &paths[12],
            })
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(None)
            })
        }
        #[cfg(target_os = "macos")]
        Some("metal-union-parallel-fp8-moe-block") if arguments.len() == 9 => {
            let manifest = PathBuf::from(&arguments[2]);
            let artifacts = PathBuf::from(&arguments[3]);
            let router = PathBuf::from(&arguments[4]);
            let kernel = PathBuf::from(&arguments[5]);
            let input = PathBuf::from(&arguments[6]);
            let reference = PathBuf::from(&arguments[7]);
            let output = PathBuf::from(&arguments[8]);
            run_metal_union_parallel_fp8_moe_block(
                &manifest, &artifacts, &router, &kernel, &input, &reference, &output,
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(None)
            })
        }
        #[cfg(target_os = "macos")]
        Some("metal-fused-gate-up-fp8-moe-block") if arguments.len() == 9 => {
            let manifest = PathBuf::from(&arguments[2]);
            let artifacts = PathBuf::from(&arguments[3]);
            let router = PathBuf::from(&arguments[4]);
            let kernel = PathBuf::from(&arguments[5]);
            let input = PathBuf::from(&arguments[6]);
            let reference = PathBuf::from(&arguments[7]);
            let output = PathBuf::from(&arguments[8]);
            run_metal_fused_gate_up_fp8_moe_block(
                &manifest, &artifacts, &router, &kernel, &input, &reference, &output,
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(None)
            })
        }
        #[cfg(target_os = "macos")]
        Some("metal-simdgroup-matrix-fp8-moe-block") if arguments.len() == 9 => {
            let manifest = PathBuf::from(&arguments[2]);
            let artifacts = PathBuf::from(&arguments[3]);
            let router = PathBuf::from(&arguments[4]);
            let kernel = PathBuf::from(&arguments[5]);
            let input = PathBuf::from(&arguments[6]);
            let reference = PathBuf::from(&arguments[7]);
            let output = PathBuf::from(&arguments[8]);
            run_metal_simdgroup_matrix_fp8_moe_block(
                &manifest, &artifacts, &router, &kernel, &input, &reference, &output,
            )
            .and_then(|report| {
                serde_json::to_writer(std::io::stdout(), &report)
                    .map_err(|error| error.to_string())?;
                println!();
                Ok(None)
            })
        }
        _ => usage(),
    };
    match result {
        Ok(Some(output)) => println!("{}", output.display()),
        Ok(None) => {}
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
    }
}
