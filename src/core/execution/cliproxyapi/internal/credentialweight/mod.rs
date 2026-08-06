// Origin: CTOX
// License: AGPL-3.0-only

pub mod weight;

#[cfg(test)]
mod weight_test;

pub use weight::{
    normalize, parse_string, parse_value, CredentialWeightError, DEFAULT_CREDENTIAL_WEIGHT,
    MAX_CREDENTIAL_WEIGHT,
};
