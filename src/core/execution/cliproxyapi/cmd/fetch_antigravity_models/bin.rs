// Origin: CTOX
// License: AGPL-3.0-only

fn main() -> std::process::ExitCode {
    ctox_cliproxyapi::internal::cmd::fetch_antigravity_models::standalone_main(
        std::env::args().skip(1),
    )
}
