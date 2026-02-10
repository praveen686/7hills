use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_root = PathBuf::from("../../contracts/proto");

    // Only compile if proto files exist (allows builds without protoc installed)
    if !proto_root.exists() {
        println!("cargo:warning=Proto directory not found at {:?}, skipping protobuf compilation", proto_root);
        return Ok(());
    }

    let proto_files: Vec<PathBuf> = [
        "quantlaxmi/v1/common.proto",
        "quantlaxmi/v1/signal.proto",
        "quantlaxmi/v1/order.proto",
        "quantlaxmi/v1/fill.proto",
        "quantlaxmi/v1/position.proto",
        "quantlaxmi/v1/market_data.proto",
        "quantlaxmi/v1/risk.proto",
        "quantlaxmi/v1/envelope.proto",
    ]
    .iter()
    .map(|p| proto_root.join(p))
    .collect();

    // Check all proto files exist
    for f in &proto_files {
        if !f.exists() {
            println!("cargo:warning=Proto file not found: {:?}, skipping", f);
            return Ok(());
        }
    }

    prost_build::Config::new()
        .out_dir("src/proto")
        .compile_protos(&proto_files, &[&proto_root])?;

    // Re-run if any proto file changes
    for f in &proto_files {
        println!("cargo:rerun-if-changed={}", f.display());
    }

    Ok(())
}
