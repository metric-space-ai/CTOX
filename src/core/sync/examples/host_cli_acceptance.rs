//! Real CTOX binary acceptance against isolated roots and localhost signaling.
#[cfg(unix)]
#[path = "support/host_cli_acceptance.rs"]
mod native;
#[cfg(unix)]
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    native::run()
}
#[cfg(not(unix))]
fn main() {
    eprintln!("native host acceptance requires a certified local listener");
    std::process::exit(2);
}
