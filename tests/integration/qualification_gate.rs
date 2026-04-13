use std::env;
use std::process::Command;

#[test]
fn model_meets_ctox_qualification_gate_when_enabled() {
    let model = match env::var("CTOX_QUALIFY_MODEL") {
        Ok(value) if !value.trim().is_empty() => value,
        _ => return,
    };
    let scenarios = env::var("CTOX_QUALIFY_SCENARIOS")
        .unwrap_or_else(|_| "minimal_ctox_stability".to_string())
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    if scenarios.is_empty() {
        return;
    }

    for scenario in scenarios {
        let mut command = Command::new("python3");
        command
            .arg("scripts/ctox_model_qualify.py")
            .arg("--model")
            .arg(&model)
            .arg("--scenario")
            .arg(&scenario);
        if let Ok(timeout) = env::var("CTOX_QUALIFY_TIMEOUT_SECS") {
            if !timeout.trim().is_empty() {
                command.arg("--timeout-secs").arg(timeout);
            }
        }
        let output = command
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .expect("failed to execute qualification runner");
        assert!(
            output.status.success(),
            "qualification failed for scenario {scenario}\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
