// ref: internal/config/disable_image_generation_mode.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::str::FromStr;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DisableImageGenerationMode {
    #[default]
    Off,
    All,
    Chat,
    Passthrough,
}

impl DisableImageGenerationMode {
    pub fn parse_json(data: &[u8]) -> Result<Self, DisableImageGenerationModeError> {
        let data = trim_ascii(data);
        if data.is_empty() || data == b"null" {
            return Ok(Self::Off);
        }
        let value: serde_json::Value = serde_json::from_slice(data)
            .map_err(|_| DisableImageGenerationModeError::InvalidValue)?;
        match value {
            serde_json::Value::Bool(false) => Ok(Self::Off),
            serde_json::Value::Bool(true) => Ok(Self::All),
            serde_json::Value::String(value) => value.parse(),
            _ => Err(DisableImageGenerationModeError::InvalidValue),
        }
    }
}

impl fmt::Display for DisableImageGenerationMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Off => "false",
            Self::All => "true",
            Self::Chat => "chat",
            Self::Passthrough => "passthrough",
        })
    }
}

impl FromStr for DisableImageGenerationMode {
    type Err = DisableImageGenerationModeError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_lowercase().as_str() {
            "" | "false" | "0" | "off" | "no" => Ok(Self::Off),
            "true" | "1" | "on" | "yes" => Ok(Self::All),
            "chat" => Ok(Self::Chat),
            "passthrough" => Ok(Self::Passthrough),
            _ => Err(DisableImageGenerationModeError::InvalidValue),
        }
    }
}

impl Serialize for DisableImageGenerationMode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Off => serializer.serialize_bool(false),
            Self::All => serializer.serialize_bool(true),
            Self::Chat => serializer.serialize_str("chat"),
            Self::Passthrough => serializer.serialize_str("passthrough"),
        }
    }
}

impl<'de> Deserialize<'de> for DisableImageGenerationMode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ModeVisitor;

        impl<'de> Visitor<'de> for ModeVisitor {
            type Value = DisableImageGenerationMode;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a boolean, null, or image-generation mode string")
            }

            fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
                Ok(if value {
                    DisableImageGenerationMode::All
                } else {
                    DisableImageGenerationMode::Off
                })
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                value.parse().map_err(E::custom)
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                self.visit_str(&value)
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(DisableImageGenerationMode::Off)
            }

            fn visit_none<E>(self) -> Result<Self::Value, E> {
                Ok(DisableImageGenerationMode::Off)
            }
        }

        deserializer.deserialize_any(ModeVisitor)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisableImageGenerationModeError {
    InvalidValue,
}

impl fmt::Display for DisableImageGenerationModeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(
            "invalid disable-image-generation value (allowed: true, false, chat, passthrough)",
        )
    }
}

impl std::error::Error for DisableImageGenerationModeError {}

fn trim_ascii(mut value: &[u8]) -> &[u8] {
    while value.first().is_some_and(u8::is_ascii_whitespace) {
        value = &value[1..];
    }
    while value.last().is_some_and(u8::is_ascii_whitespace) {
        value = &value[..value.len() - 1];
    }
    value
}
