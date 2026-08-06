// ref: internal/misc/oauth.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::error::Error;
use std::fmt;
use std::sync::mpsc::{self, Receiver};
use std::thread;

use url::Url;

/// Failure to obtain cryptographically secure operating-system randomness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RandomStateError;

impl fmt::Display for RandomStateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("failed to generate random bytes")
    }
}

impl Error for RandomStateError {}

/// Generates the 16 cryptographically random bytes and lowercase hexadecimal
/// encoding used by the upstream OAuth state parameter.
pub fn generate_random_state() -> Result<String, RandomStateError> {
    let mut bytes = [0_u8; 16];
    getrandom::fill(&mut bytes).map_err(|_| RandomStateError)?;

    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut state = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        state.push(HEX[(byte >> 4) as usize] as char);
        state.push(HEX[(byte & 0x0f) as usize] as char);
    }
    Ok(state)
}

/// Parsed OAuth callback parameters.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OAuthCallback {
    pub code: String,
    pub state: String,
    pub error: String,
    pub error_description: String,
}

/// Error returned when an OAuth callback is not a URL-shaped callback or does
/// not contain either an authorization code or an OAuth error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParseOAuthCallbackError {
    InvalidCallbackUrl,
    Url(url::ParseError),
    MissingCode,
}

impl fmt::Display for ParseOAuthCallbackError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCallbackUrl => formatter.write_str("invalid callback URL"),
            Self::Url(error) => error.fmt(formatter),
            Self::MissingCode => formatter.write_str("callback URL missing code"),
        }
    }
}

impl Error for ParseOAuthCallbackError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Url(error) => Some(error),
            Self::InvalidCallbackUrl | Self::MissingCode => None,
        }
    }
}

/// Runs a prompt on a detached thread and returns independent, capacity-one
/// result and error receivers. A dropped receiver never prevents the worker
/// thread from completing.
pub fn async_prompt<F, E>(prompt_fn: F, message: String) -> (Receiver<String>, Receiver<E>)
where
    F: FnOnce(String) -> Result<String, E> + Send + 'static,
    E: Send + 'static,
{
    let (input_sender, input_receiver) = mpsc::sync_channel(1);
    let (error_sender, error_receiver) = mpsc::sync_channel(1);
    thread::spawn(move || match prompt_fn(message) {
        Ok(input) => {
            let _ = input_sender.send(input);
        }
        Err(error) => {
            let _ = error_sender.send(error);
        }
    });
    (input_receiver, error_receiver)
}

/// Extracts OAuth parameters from a callback URL. Blank input maps to `None`,
/// preserving the Go function's nil result.
pub fn parse_oauth_callback(input: &str) -> Result<Option<OAuthCallback>, ParseOAuthCallbackError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    let candidate = normalize_callback_candidate(trimmed)?;
    let parsed = Url::parse(&candidate).map_err(ParseOAuthCallbackError::Url)?;

    let mut callback = OAuthCallback {
        code: query_value(parsed.query(), "code"),
        state: query_value(parsed.query(), "state"),
        error: query_value(parsed.query(), "error"),
        error_description: query_value(parsed.query(), "error_description"),
    };

    if let Some(fragment) = parsed.fragment() {
        // Go ignores all fragment values when ParseQuery reports an error.
        if valid_form_query(fragment) {
            fill_if_blank(&mut callback.code, query_value(Some(fragment), "code"));
            fill_if_blank(&mut callback.state, query_value(Some(fragment), "state"));
            fill_if_blank(&mut callback.error, query_value(Some(fragment), "error"));
            fill_if_blank(
                &mut callback.error_description,
                query_value(Some(fragment), "error_description"),
            );
        }
    }

    if !callback.code.is_empty() && callback.state.is_empty() {
        if let Some((code, state)) = callback.code.split_once('#') {
            callback.state = state.to_owned();
            callback.code = code.to_owned();
        }
    }

    if callback.error.is_empty() && !callback.error_description.is_empty() {
        callback.error = std::mem::take(&mut callback.error_description);
    }

    if callback.code.is_empty() && callback.error.is_empty() {
        return Err(ParseOAuthCallbackError::MissingCode);
    }

    Ok(Some(callback))
}

fn normalize_callback_candidate(trimmed: &str) -> Result<String, ParseOAuthCallbackError> {
    if trimmed.contains("://") {
        return Ok(trimmed.to_owned());
    }
    if trimmed.starts_with('?') {
        return Ok(format!("http://localhost{trimmed}"));
    }
    if trimmed.contains(['/', '?', '#']) || trimmed.contains(':') {
        return Ok(format!("http://{trimmed}"));
    }
    if trimmed.contains('=') {
        return Ok(format!("http://localhost/?{trimmed}"));
    }
    Err(ParseOAuthCallbackError::InvalidCallbackUrl)
}

fn query_value(query: Option<&str>, key: &str) -> String {
    let Some(query) = query else {
        return String::new();
    };

    query
        .split('&')
        .filter(|pair| valid_form_pair(pair))
        .filter_map(|pair| url::form_urlencoded::parse(pair.as_bytes()).next())
        .find_map(|(name, value)| (name == key).then(|| value.trim().to_owned()))
        .unwrap_or_default()
}

fn fill_if_blank(target: &mut String, value: String) {
    if target.is_empty() {
        *target = value;
    }
}

fn valid_form_query(query: &str) -> bool {
    query.split('&').all(valid_form_pair)
}

fn valid_form_pair(pair: &str) -> bool {
    !pair.contains(';')
        && !pair
            .bytes()
            .enumerate()
            .any(|(index, byte)| byte == b'%' && !has_hex_escape(pair.as_bytes(), index))
}

fn has_hex_escape(bytes: &[u8], percent: usize) -> bool {
    bytes
        .get(percent + 1..percent + 3)
        .is_some_and(|digits| digits.iter().all(u8::is_ascii_hexdigit))
}
