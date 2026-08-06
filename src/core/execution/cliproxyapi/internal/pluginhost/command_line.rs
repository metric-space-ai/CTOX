// ref: internal/pluginhost/command_line.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: typed flag plans and injected auth/output authority replace ambient flag/env/stdout use
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::sdk::pluginapi::{
    AuthData, CommandLineExecutionRequest, CommandLineFlag, CommandLineFlagValue,
    CommandLinePlugin, CommandLineRegistrationRequest, HostConfigSummary, Metadata,
};

pub trait CommandLineAuthSink: Send + Sync {
    fn save(&self, auth: &AuthData) -> Result<String, String>;
}

#[derive(Clone)]
pub struct CommandLinePluginRecord {
    pub plugin_id: String,
    pub priority: i32,
    pub metadata: Metadata,
    pub plugin: Arc<dyn CommandLinePlugin>,
}

#[derive(Clone, Debug)]
struct FlagRecord {
    plugin_id: String,
    priority: i32,
    flag: CommandLineFlag,
    value: String,
    set: bool,
}

#[derive(Clone)]
struct RegisteredPlugin {
    metadata: Metadata,
    plugin: Arc<dyn CommandLinePlugin>,
}

#[derive(Default)]
pub struct CommandLineRegistry {
    flags: BTreeMap<String, FlagRecord>,
    plugins: BTreeMap<String, RegisteredPlugin>,
    triggered: BTreeSet<String>,
}

impl CommandLineRegistry {
    pub async fn register(
        &mut self,
        records: &[CommandLinePluginRecord],
        native_flags: &BTreeSet<String>,
    ) -> Vec<CommandLineRegistrationError> {
        let mut errors = Vec::new();
        for record in records {
            let response = match record
                .plugin
                .register_command_line(CommandLineRegistrationRequest {
                    plugin: record.metadata.clone(),
                })
                .await
            {
                Ok(response) => response,
                Err(error) => {
                    errors.push(CommandLineRegistrationError {
                        plugin_id: record.plugin_id.clone(),
                        flag: None,
                        reason: error.to_string(),
                    });
                    continue;
                }
            };
            self.plugins.insert(
                record.plugin_id.clone(),
                RegisteredPlugin {
                    metadata: record.metadata.clone(),
                    plugin: record.plugin.clone(),
                },
            );
            for flag in response.flags {
                let name = flag.name.trim().to_owned();
                let kind = normalize_flag_kind(&flag.kind);
                let normalized = kind
                    .as_deref()
                    .and_then(|kind| normalize_flag_value(kind, &flag.default_value));
                let conflict = native_flags.contains(&name)
                    || self
                        .flags
                        .get(&name)
                        .is_some_and(|existing| existing.priority >= record.priority);
                if !valid_flag_name(&name) || kind.is_none() || normalized.is_none() || conflict {
                    errors.push(CommandLineRegistrationError {
                        plugin_id: record.plugin_id.clone(),
                        flag: Some(name),
                        reason: "invalid, unsupported, or conflicting flag".to_owned(),
                    });
                    continue;
                }
                let kind = kind.expect("checked above");
                let value = normalized.expect("checked above");
                self.flags.insert(
                    name.clone(),
                    FlagRecord {
                        plugin_id: record.plugin_id.clone(),
                        priority: record.priority,
                        flag: CommandLineFlag {
                            name,
                            kind,
                            default_value: value.clone(),
                            ..flag
                        },
                        value,
                        set: false,
                    },
                );
            }
        }
        errors
    }

    pub fn set(&mut self, name: &str, raw: &str) -> Result<(), CommandLineError> {
        let record = self
            .flags
            .get_mut(name)
            .ok_or(CommandLineError::UnknownFlag)?;
        record.value =
            normalize_flag_value(&record.flag.kind, raw).ok_or(CommandLineError::InvalidValue)?;
        record.set = true;
        self.triggered.insert(name.to_owned());
        Ok(())
    }

    pub fn has_triggered_flags(&self) -> bool {
        !self.triggered.is_empty()
    }

    pub async fn execute(
        &self,
        program: String,
        args: Vec<String>,
        config_path: String,
        host: HostConfigSummary,
        native_flags: BTreeMap<String, CommandLineFlagValue>,
        auth_sink: &dyn CommandLineAuthSink,
    ) -> CommandLineOutcome {
        let mut all_flags = native_flags;
        let mut triggered_by_plugin =
            BTreeMap::<String, BTreeMap<String, CommandLineFlagValue>>::new();
        for (name, record) in &self.flags {
            let value = CommandLineFlagValue {
                name: name.clone(),
                kind: record.flag.kind.clone(),
                value: record.value.clone(),
                set: record.set,
            };
            all_flags.insert(name.clone(), value.clone());
            if self.triggered.contains(name) {
                triggered_by_plugin
                    .entry(record.plugin_id.clone())
                    .or_default()
                    .insert(name.clone(), value);
            }
        }
        let mut outcome = CommandLineOutcome::default();
        for (plugin_id, triggered_flags) in triggered_by_plugin {
            let Some(plugin) = self.plugins.get(&plugin_id) else {
                continue;
            };
            outcome.handled = true;
            let response = plugin
                .plugin
                .execute_command_line(CommandLineExecutionRequest {
                    plugin: plugin.metadata.clone(),
                    program: program.clone(),
                    args: args.clone(),
                    config_path: config_path.clone(),
                    host: host.clone(),
                    flags: all_flags.clone(),
                    triggered_flags,
                })
                .await;
            match response {
                Ok(response) => {
                    outcome.stdout.extend(response.stdout);
                    outcome.stderr.extend(response.stderr);
                    if response.exit_code == 0 {
                        for auth in &response.auths {
                            match auth_sink.save(auth) {
                                Ok(path) if !path.trim().is_empty() => {
                                    append_saved_path(&mut outcome.stdout, &path);
                                    outcome.saved_paths.push(path);
                                }
                                Ok(_) => {}
                                Err(error) => {
                                    append_line(&mut outcome.stderr, &error);
                                    outcome.exit_code = outcome.exit_code.max(1);
                                }
                            }
                        }
                    }
                    if outcome.exit_code == 0 && response.exit_code != 0 {
                        outcome.exit_code = response.exit_code;
                    }
                }
                Err(error) => {
                    append_line(&mut outcome.stderr, &error.to_string());
                    outcome.exit_code = outcome.exit_code.max(1);
                }
            }
        }
        outcome
    }
}

fn valid_flag_name(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('-')
        && !matches!(name, "help" | "h")
        && !name
            .chars()
            .any(|character| character.is_whitespace() || character == '=')
}

fn normalize_flag_kind(kind: &str) -> Option<String> {
    let kind = kind.trim().to_ascii_lowercase();
    match kind.as_str() {
        "" | "bool" => Some("bool".to_owned()),
        "string" | "int" | "int64" | "float64" | "duration" => Some(kind),
        _ => None,
    }
}

fn normalize_flag_value(kind: &str, value: &str) -> Option<String> {
    let trimmed = value.trim();
    match kind {
        "bool" => match trimmed {
            "" => Some("false".to_owned()),
            "true" | "1" | "t" | "TRUE" => Some("true".to_owned()),
            "false" | "0" | "f" | "FALSE" => Some("false".to_owned()),
            _ => None,
        },
        "string" => Some(value.to_owned()),
        "int" | "int64" => trimmed.parse::<i64>().ok().map(|value| value.to_string()),
        "float64" => trimmed.parse::<f64>().ok().map(|value| value.to_string()),
        "duration" => normalize_duration(trimmed),
        _ => None,
    }
}

fn normalize_duration(value: &str) -> Option<String> {
    if value.is_empty() {
        return Some("0s".to_owned());
    }
    let split = value.find(|character: char| {
        !character.is_ascii_digit() && character != '.' && character != '-'
    })?;
    let (number, unit) = value.split_at(split);
    number.parse::<f64>().ok()?;
    matches!(unit, "ns" | "us" | "µs" | "ms" | "s" | "m" | "h").then(|| value.to_owned())
}

fn append_saved_path(output: &mut Vec<u8>, path: &str) {
    if !output.is_empty() && !output.ends_with(b"\n") {
        output.push(b'\n');
    }
    output.extend(format!("Authentication saved to {path}\n").as_bytes());
}

fn append_line(output: &mut Vec<u8>, line: &str) {
    output.extend(line.as_bytes());
    if !output.ends_with(b"\n") {
        output.push(b'\n');
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommandLineRegistrationError {
    pub plugin_id: String,
    pub flag: Option<String>,
    pub reason: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommandLineError {
    UnknownFlag,
    InvalidValue,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CommandLineOutcome {
    pub exit_code: i32,
    pub handled: bool,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub saved_paths: Vec<String>,
}
