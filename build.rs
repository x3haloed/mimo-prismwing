fn main() {
    println!("cargo:rerun-if-changed=src/pytorch_topk.cpp");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        cc::Build::new()
            .cpp(true)
            .file("src/pytorch_topk.cpp")
            .flag("-std=c++17")
            .compile("pytorch_topk");
    }
}
