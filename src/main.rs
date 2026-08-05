use prismwing::{build_census, write_census};
use std::path::PathBuf;

fn usage() -> ! {
    eprintln!(
        "usage: prismwing census <model.safetensors.index.json> <checkpoint-dir> <output.json>"
    );
    std::process::exit(2);
}

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    if arguments.len() != 5 || arguments[1] != "census" {
        usage();
    }
    let index = PathBuf::from(&arguments[2]);
    let checkpoint = PathBuf::from(&arguments[3]);
    let output = PathBuf::from(&arguments[4]);
    match build_census(&index, &checkpoint).and_then(|census| write_census(&census, &output)) {
        Ok(()) => println!("{}", output.display()),
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
    }
}
