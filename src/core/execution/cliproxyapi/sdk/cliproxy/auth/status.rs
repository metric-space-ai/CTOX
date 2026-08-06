// ref: sdk/cliproxy/auth/status.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Lifecycle state of an auth entry. `Other` preserves forward-compatible
/// upstream strings because Go's named string type is not a closed enum.
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub enum AuthStatus {
    #[default]
    Unknown,
    Active,
    Pending,
    Refreshing,
    Error,
    Disabled,
    Other(String),
}

impl AuthStatus {
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Unknown => "unknown",
            Self::Active => "active",
            Self::Pending => "pending",
            Self::Refreshing => "refreshing",
            Self::Error => "error",
            Self::Disabled => "disabled",
            Self::Other(value) => value,
        }
    }

    #[must_use]
    pub fn is_known(&self) -> bool {
        !matches!(self, Self::Other(_))
    }
}

impl From<&str> for AuthStatus {
    fn from(value: &str) -> Self {
        match value {
            "unknown" => Self::Unknown,
            "active" => Self::Active,
            "pending" => Self::Pending,
            "refreshing" => Self::Refreshing,
            "error" => Self::Error,
            "disabled" => Self::Disabled,
            other => Self::Other(other.to_owned()),
        }
    }
}

impl From<String> for AuthStatus {
    fn from(value: String) -> Self {
        match value.as_str() {
            "unknown" => Self::Unknown,
            "active" => Self::Active,
            "pending" => Self::Pending,
            "refreshing" => Self::Refreshing,
            "error" => Self::Error,
            "disabled" => Self::Disabled,
            _ => Self::Other(value),
        }
    }
}

impl fmt::Display for AuthStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl Serialize for AuthStatus {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for AuthStatus {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        String::deserialize(deserializer).map(Self::from)
    }
}
