use serde::Deserialize;
use serde::Serialize;
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GpuLiveState {
    pub index: usize,
    pub uuid: Option<String>,
    pub name: String,
    pub total_mb: u64,
    pub used_mb: u64,
    pub free_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResourceSnapshot {
    pub gpus: Vec<GpuLiveState>,
    pub source: String,
}

impl ResourceSnapshot {
    pub fn gpu(&self, index: usize) -> Option<&GpuLiveState> {
        self.gpus.iter().find(|gpu| gpu.index == index)
    }
}

pub fn inspect_resource_snapshot() -> Option<ResourceSnapshot> {
    if let Ok(override_json) = std::env::var("CTOX_RESOURCE_SNAPSHOT_JSON") {
        if !override_json.trim().is_empty() {
            return serde_json::from_str(&override_json).ok();
        }
    }
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let mut gpus = Vec::new();
    for line in stdout.lines() {
        let parts = line
            .split(',')
            .map(|chunk| chunk.trim())
            .collect::<Vec<_>>();
        if parts.len() < 6 {
            continue;
        }
        let Ok(index) = parts[0].parse::<usize>() else {
            continue;
        };
        let Ok(total_mb) = parts[3].parse::<u64>() else {
            continue;
        };
        let Ok(used_mb) = parts[4].parse::<u64>() else {
            continue;
        };
        let Ok(free_mb) = parts[5].parse::<u64>() else {
            continue;
        };
        gpus.push(GpuLiveState {
            index,
            uuid: Some(parts[1].to_string()).filter(|value| !value.is_empty()),
            name: parts[2].to_string(),
            total_mb,
            used_mb,
            free_mb,
        });
    }
    if gpus.is_empty() {
        return None;
    }
    Some(ResourceSnapshot {
        gpus,
        source: "nvidia-smi".to_string(),
    })
}
