use prismwing::{
    build_census, inspect_mapped_tensor, repack_expert_container, verify_expert_container,
    write_census,
};
use std::path::PathBuf;

fn usage() -> ! {
    eprintln!("usage:");
    eprintln!("  prismwing census <model.safetensors.index.json> <checkpoint-dir> <output.json>");
    eprintln!("  prismwing repack <source.safetensors> <output.pwexpert> <tensor> [tensor ...]");
    eprintln!("  prismwing verify-container <container.pwexpert>");
    eprintln!("  prismwing inspect-tensor <source.safetensors> <tensor>");
    std::process::exit(2);
}

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    let result: Result<Option<PathBuf>, String> = match arguments.get(1).map(String::as_str) {
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
