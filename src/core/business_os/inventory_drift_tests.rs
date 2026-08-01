// SG1: the command-type inventory drift check runs in `cargo test`, not only
// as a tool somebody remembers to invoke.
//
// This exists because of a failure it would have caught. S-CUT4f moved
// accept_rxdb_business_command_with_origin from store.rs into command_plane.rs.
// The generator located that function by string index in store.rs, so it began
// throwing "cannot locate authoritative Business OS command classifier" — and
// nothing noticed, because the drift check was a separate command nobody ran
// during the refactor. The inventory that store.rs include_str!'s at build time
// silently went stale.
//
// A contract check that lives outside the test suite is a contract check that
// only holds while someone remembers it.

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::process::Command;

    #[test]
    fn business_command_inventory_matches_the_dispatcher() {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let tool = repo.join("src/core/business_os/tools/build_business_command_inventory.mjs");
        assert!(
            tool.is_file(),
            "inventory generator missing at {}",
            tool.display()
        );

        let output = match Command::new("node").arg(&tool).arg("--check").output() {
            Ok(output) => output,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                // Node is the generator's runtime. Skipping is honest here —
                // failing would report a missing toolchain as inventory drift,
                // which is a different thing and would teach people to ignore
                // this test.
                eprintln!("skipping inventory drift check: node not on PATH");
                return;
            }
            Err(err) => panic!("failed to run inventory generator: {err}"),
        };

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "the command-type inventory no longer matches the dispatcher.\n\
             Regenerate it with:\n  \
             node src/core/business_os/tools/build_business_command_inventory.mjs\n\
             and commit the result together with the change that moved a command type.\n\
             \n--- generator stdout ---\n{stdout}\n--- generator stderr ---\n{stderr}"
        );
    }
}
