//! Server configuration options

use clap::Args;
use clap::ValueEnum;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ServerTransport {
    LocalIpc,
}

impl Default for ServerTransport {
    fn default() -> Self {
        Self::LocalIpc
    }
}

/// Local server configuration
#[derive(Args, Clone, Deserialize)]
pub struct ServerOptions {
    /// Local IPC endpoint for direct in-process Responses streaming
    #[arg(long)]
    #[serde(default)]
    pub transport_endpoint: Option<PathBuf>,

    /// Server transport contract
    #[arg(long, value_enum, default_value_t = ServerTransport::LocalIpc)]
    #[serde(default)]
    pub transport: ServerTransport,

    /// MCP protocol server port (enables MCP if set)
    #[arg(long)]
    #[serde(default)]
    pub mcp_port: Option<u16>,

    /// MCP client configuration file path
    #[arg(long)]
    #[serde(default)]
    pub mcp_config: Option<PathBuf>,
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            transport_endpoint: None,
            transport: ServerTransport::LocalIpc,
            mcp_port: None,
            mcp_config: None,
        }
    }
}
