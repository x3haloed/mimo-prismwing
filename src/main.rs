use prismwing::{build_census, repack_expert_container, verify_expert_container, write_census};
use std::path::PathBuf;

fn usage() -> ! {
    eprintln!("usage:");
    eprintln!("  prismwing census <model.safetensors.index.json> <checkpoint-dir> <output.json>");
    eprintln!("  prismwing repack <source.safetensors> <output.pwexpert> <tensor> [tensor ...]");
    eprintln!("  prismwing verify-container <container.pwexpert>");
    std::process::exit(2);
}

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    let result = match arguments.get(1).map(String::as_str) {
        Some("census") if arguments.len() == 5 => {
            let index = PathBuf::from(&arguments[2]);
            let checkpoint = PathBuf::from(&arguments[3]);
            let output = PathBuf::from(&arguments[4]);
            build_census(&index, &checkpoint)
                .and_then(|census| write_census(&census, &output))
                .map(|_| output)
        }
        Some("repack") if arguments.len() >= 5 => {
            let source = PathBuf::from(&arguments[2]);
            let output = PathBuf::from(&arguments[3]);
            repack_expert_container(&source, &output, &arguments[4..]).map(|_| output)
        }
        Some("verify-container") if arguments.len() == 3 => {
            let container = PathBuf::from(&arguments[2]);
            verify_expert_container(&container).map(|_| container)
        }
        _ => usage(),
    };
    match result {
        Ok(output) => println!("{}", output.display()),
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
    }
}
