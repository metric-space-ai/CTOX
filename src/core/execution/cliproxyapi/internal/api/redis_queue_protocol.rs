// ref: internal/api/redis_queue_protocol.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::sync::Arc;

const MAX_RESP_LINE: usize = 16 * 1024;
const MAX_RESP_ITEMS: usize = 1_024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RespValue {
    Simple(String),
    Error(String),
    Integer(i64),
    Bulk(Option<Vec<u8>>),
    Array(Vec<RespValue>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RespError {
    Incomplete,
    Invalid,
    TooLarge,
}

impl fmt::Display for RespError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Incomplete => "incomplete RESP frame",
            Self::Invalid => "invalid RESP frame",
            Self::TooLarge => "RESP frame exceeds limit",
        })
    }
}
impl std::error::Error for RespError {}

pub fn decode_resp(input: &[u8]) -> Result<(RespValue, usize), RespError> {
    parse_value(input, 0)
}

fn parse_value(input: &[u8], depth: usize) -> Result<(RespValue, usize), RespError> {
    if depth > 8 {
        return Err(RespError::TooLarge);
    }
    let prefix = *input.first().ok_or(RespError::Incomplete)?;
    let (line, consumed) = read_line(input.get(1..).ok_or(RespError::Incomplete)?)?;
    let head = consumed + 1;
    match prefix {
        b'+' => Ok((RespValue::Simple(text(line)?), head)),
        b'-' => Ok((RespValue::Error(text(line)?), head)),
        b':' => Ok((RespValue::Integer(number(line)?), head)),
        b'$' => {
            let len = number(line)?;
            if len == -1 {
                return Ok((RespValue::Bulk(None), head));
            }
            let len = usize::try_from(len).map_err(|_| RespError::Invalid)?;
            if len > MAX_RESP_LINE {
                return Err(RespError::TooLarge);
            }
            let end = head.checked_add(len).ok_or(RespError::TooLarge)?;
            if input.get(end..end + 2) != Some(b"\r\n") {
                return Err(RespError::Incomplete);
            }
            Ok((RespValue::Bulk(Some(input[head..end].to_vec())), end + 2))
        }
        b'*' => {
            let count = usize::try_from(number(line)?).map_err(|_| RespError::Invalid)?;
            if count > MAX_RESP_ITEMS {
                return Err(RespError::TooLarge);
            }
            let mut values = Vec::with_capacity(count);
            let mut offset = head;
            for _ in 0..count {
                let (value, used) =
                    parse_value(input.get(offset..).ok_or(RespError::Incomplete)?, depth + 1)?;
                offset = offset.checked_add(used).ok_or(RespError::TooLarge)?;
                values.push(value);
            }
            Ok((RespValue::Array(values), offset))
        }
        _ => Err(RespError::Invalid),
    }
}

fn read_line(input: &[u8]) -> Result<(&[u8], usize), RespError> {
    let end = input
        .windows(2)
        .position(|window| window == b"\r\n")
        .ok_or(RespError::Incomplete)?;
    if end > MAX_RESP_LINE {
        return Err(RespError::TooLarge);
    }
    Ok((&input[..end], end + 2))
}
fn text(raw: &[u8]) -> Result<String, RespError> {
    std::str::from_utf8(raw)
        .map(str::to_owned)
        .map_err(|_| RespError::Invalid)
}
fn number(raw: &[u8]) -> Result<i64, RespError> {
    std::str::from_utf8(raw)
        .map_err(|_| RespError::Invalid)?
        .parse()
        .map_err(|_| RespError::Invalid)
}

#[must_use]
pub fn encode_resp(value: &RespValue) -> Vec<u8> {
    let mut out = Vec::new();
    encode_into(value, &mut out);
    out
}
fn encode_into(value: &RespValue, out: &mut Vec<u8>) {
    match value {
        RespValue::Simple(value) => {
            out.extend_from_slice(b"+");
            out.extend_from_slice(value.as_bytes());
            out.extend_from_slice(b"\r\n");
        }
        RespValue::Error(value) => {
            out.extend_from_slice(b"-");
            out.extend_from_slice(value.as_bytes());
            out.extend_from_slice(b"\r\n");
        }
        RespValue::Integer(value) => out.extend_from_slice(format!(":{value}\r\n").as_bytes()),
        RespValue::Bulk(None) => out.extend_from_slice(b"$-1\r\n"),
        RespValue::Bulk(Some(value)) => {
            out.extend_from_slice(format!("${}\r\n", value.len()).as_bytes());
            out.extend_from_slice(value);
            out.extend_from_slice(b"\r\n");
        }
        RespValue::Array(values) => {
            out.extend_from_slice(format!("*{}\r\n", values.len()).as_bytes());
            for value in values {
                encode_into(value, out);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RedisQueueError {
    Authentication,
    Read,
    UnsupportedChannel,
}

pub trait RedisQueueAuthority: Send + Sync {
    fn authenticate(&self, password: &str) -> bool;
    fn pop(&self, channel: &str, count: usize) -> Result<Vec<Vec<u8>>, RedisQueueError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RedisQueuePolicy {
    pub management_enabled: bool,
    pub home_enabled: bool,
    pub local_client: bool,
}

pub struct RedisQueueSession {
    authority: Arc<dyn RedisQueueAuthority>,
    policy: RedisQueuePolicy,
    authenticated: bool,
}

impl RedisQueueSession {
    #[must_use]
    pub fn new(authority: Arc<dyn RedisQueueAuthority>, policy: RedisQueuePolicy) -> Self {
        Self {
            authority,
            authenticated: policy.local_client,
            policy,
        }
    }

    pub fn handle(&mut self, frame: &RespValue) -> Vec<RespValue> {
        if !self.policy.management_enabled {
            return vec![error("redis usage output disabled")];
        }
        if self.policy.home_enabled {
            return vec![error("redis usage output disabled in home mode")];
        }
        let Some(args) = command_args(frame) else {
            return vec![error("invalid command")];
        };
        let command = args
            .first()
            .map(|value| value.to_ascii_uppercase())
            .unwrap_or_default();
        if command == "AUTH" {
            let password = match args.as_slice() {
                [_, password] => password,
                [_, _, password] => password,
                _ => return vec![error("wrong number of arguments for AUTH")],
            };
            self.authenticated = self.authority.authenticate(password);
            return vec![if self.authenticated {
                RespValue::Simple("OK".into())
            } else {
                error("invalid password")
            }];
        }
        if !self.authenticated {
            return vec![error("NOAUTH authentication required")];
        }
        match command.as_str() {
            "PING" => vec![args.get(1).map_or_else(
                || RespValue::Simple("PONG".into()),
                |value| bulk(value.as_bytes()),
            )],
            "LPOP" | "RPOP" => self.handle_pop(&args),
            "SUBSCRIBE" => self.handle_subscribe(&args),
            _ => vec![error("unsupported command")],
        }
    }

    fn handle_pop(&self, args: &[String]) -> Vec<RespValue> {
        if !(args.len() == 2 || args.len() == 3) {
            return vec![error("wrong number of arguments for POP")];
        }
        let count = if let Some(raw) = args.get(2) {
            match raw.parse::<usize>() {
                Ok(0) | Err(_) => return vec![error("invalid count")],
                Ok(count) => count.min(1_024),
            }
        } else {
            1
        };
        match self.authority.pop(&args[1], count) {
            Ok(items) if args.len() == 2 => {
                vec![items.first().map_or(RespValue::Bulk(None), |item| {
                    RespValue::Bulk(Some(item.clone()))
                })]
            }
            Ok(items) => vec![RespValue::Array(
                items
                    .into_iter()
                    .map(|item| RespValue::Bulk(Some(item)))
                    .collect(),
            )],
            Err(RedisQueueError::UnsupportedChannel) => {
                vec![error(&format!("unsupported channel '{}'", args[1]))]
            }
            Err(_) => vec![error("queue unavailable")],
        }
    }

    fn handle_subscribe(&self, args: &[String]) -> Vec<RespValue> {
        let Some(channel) = args.get(1).filter(|_| args.len() == 2) else {
            return vec![error("wrong number of arguments for SUBSCRIBE")];
        };
        if channel != "usage" && channel != "errors" {
            return vec![error(&format!("unsupported channel '{channel}'"))];
        }
        let mut responses = vec![RespValue::Array(vec![
            bulk(b"subscribe"),
            bulk(channel.as_bytes()),
            RespValue::Integer(1),
        ])];
        if channel == "usage" {
            responses.push(pubsub_message(channel, b"{\"support_refresh\":true}"));
        }
        responses
    }
}

fn command_args(frame: &RespValue) -> Option<Vec<String>> {
    let RespValue::Array(values) = frame else {
        return None;
    };
    values
        .iter()
        .map(|value| match value {
            RespValue::Bulk(Some(raw)) => std::str::from_utf8(raw).ok().map(str::to_owned),
            RespValue::Simple(value) => Some(value.clone()),
            _ => None,
        })
        .collect()
}
fn bulk(value: &[u8]) -> RespValue {
    RespValue::Bulk(Some(value.to_vec()))
}
fn error(value: &str) -> RespValue {
    RespValue::Error(format!("ERR {value}"))
}
fn pubsub_message(channel: &str, payload: &[u8]) -> RespValue {
    RespValue::Array(vec![
        bulk(b"message"),
        bulk(channel.as_bytes()),
        bulk(payload),
    ])
}

impl fmt::Debug for RedisQueueSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RedisQueueSession")
            .field("policy", &self.policy)
            .field("authenticated", &self.authenticated)
            .finish_non_exhaustive()
    }
}
