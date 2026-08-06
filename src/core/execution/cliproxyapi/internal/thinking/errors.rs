// ref: internal/thinking/errors.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::{collections::BTreeMap, error::Error, fmt};

use serde_json::Value;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ErrorCode {
    InvalidSuffix,
    UnknownLevel,
    ThinkingNotSupported,
    LevelNotSupported,
    BudgetOutOfRange,
    ProviderMismatch,
    Unknown(String),
}

impl ErrorCode {
    pub fn as_str(&self) -> &str {
        match self {
            Self::InvalidSuffix => "INVALID_SUFFIX",
            Self::UnknownLevel => "UNKNOWN_LEVEL",
            Self::ThinkingNotSupported => "THINKING_NOT_SUPPORTED",
            Self::LevelNotSupported => "LEVEL_NOT_SUPPORTED",
            Self::BudgetOutOfRange => "BUDGET_OUT_OF_RANGE",
            Self::ProviderMismatch => "PROVIDER_MISMATCH",
            Self::Unknown(value) => value,
        }
    }
}

impl From<&str> for ErrorCode {
    fn from(value: &str) -> Self {
        match value {
            "INVALID_SUFFIX" => Self::InvalidSuffix,
            "UNKNOWN_LEVEL" => Self::UnknownLevel,
            "THINKING_NOT_SUPPORTED" => Self::ThinkingNotSupported,
            "LEVEL_NOT_SUPPORTED" => Self::LevelNotSupported,
            "BUDGET_OUT_OF_RANGE" => Self::BudgetOutOfRange,
            "PROVIDER_MISMATCH" => Self::ProviderMismatch,
            _ => Self::Unknown(value.to_owned()),
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ThinkingError {
    pub code: ErrorCode,
    pub message: String,
    pub model: String,
    pub details: Option<BTreeMap<String, Value>>,
}

impl ThinkingError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            model: String::new(),
            details: None,
        }
    }

    pub fn with_model(
        code: ErrorCode,
        message: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            code,
            message: message.into(),
            model: model.into(),
            details: None,
        }
    }

    /// Portable HTTP status used by the Go error and the Rust handlers.
    pub const fn status_code(&self) -> u16 {
        400
    }
}

impl fmt::Display for ThinkingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The upstream Error method deliberately exposes only Message.
        formatter.write_str(&self.message)
    }
}

impl Error for ThinkingError {}
