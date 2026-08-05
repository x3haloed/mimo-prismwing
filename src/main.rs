use prismwing::{
    build_census, inspect_mapped_tensor, repack_expert_container, run_mapped_fp8_gemv,
    verify_expert_container, write_census,
};
#[cfg(target_os = "macos")]
use prismwing::{
    run_metal_dynamic_fp8_moe_block, run_metal_fp8_expert, run_metal_fp8_expert_batch8,
    run_metal_fp8_expert_batch8_shared_weight, run_metal_fp8_moe_block,
    run_metal_fused_gate_up_fp8_moe_block, run_metal_mapped_fp8_gemv, run_metal_noaux_tc_router,
    run_metal_simdgroup_matrix_fp8_moe_block, run_metal_union_parallel_fp8_moe_block,
};
use std::path::PathBuf;

fn usage() -> ! {
    eprintln!("usage:");
    eprintln!("  prismwing census <model.safetensors.index.json> <checkpoint-dir> <output.json>");
    eprintln!("  prismwing repack <source.safetensors> <output.pwexpert> <tensor> [tensor ...]");
    eprintln!("  prismwing verify-container <container.pwexpert>");
    eprintln!("  prismwing inspect-tensor <source.safetensors> <tensor>");
    eprintln!(
        "  prismwing fp8-gemv <source.safetensors> <weight> <scale> <input.f32> <output.f32>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing metal-fp8-gemv <source.safetensors> <kernel.metal> <weight> <scale> <input.f32> <reference.f32> <output.f32>"
    );
    #[cfg(target_os = "macos")]
    eprintln!(
        "  prismwing metal-fp8-expert <gate-up.safetensors> <down.safetensors> <kernel.metal> <input.f32> <reference.f32> <output.f32>"
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
    std::process::exit(2);
}

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    let result: Result<Option<PathBuf>, String> =
        match arguments.get(1).map(String::as_str) {
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
                run_mapped_fp8_gemv(&source, &arguments[3], &arguments[4], &input, &output)
                    .and_then(|report| {
                        serde_json::to_writer(std::io::stdout(), &report)
                            .map_err(|error| error.to_string())?;
                        println!();
                        Ok(None)
                    })
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
            Some("metal-fp8-expert") if arguments.len() == 8 => {
                let gate_up = PathBuf::from(&arguments[2]);
                let down = PathBuf::from(&arguments[3]);
                let kernel = PathBuf::from(&arguments[4]);
                let input = PathBuf::from(&arguments[5]);
                let reference = PathBuf::from(&arguments[6]);
                let output = PathBuf::from(&arguments[7]);
                run_metal_fp8_expert(&gate_up, &down, &kernel, &input, &reference, &output)
                    .and_then(|report| {
                        serde_json::to_writer(std::io::stdout(), &report)
                            .map_err(|error| error.to_string())?;
                        println!();
                        Ok(None)
                    })
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
