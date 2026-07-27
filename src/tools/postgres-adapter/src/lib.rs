// Origin: CTOX
// License: AGPL-3.0-only

use anyhow::{bail, Context, Result};
use base64::Engine;
use bytes::BytesMut;
use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, SecondsFormat, Utc};
pub use ctox_sqlserver_adapter::{validate_read_statement, validate_write_statement, SqlParameter};
use postgres_native_tls::MakeTlsConnector;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Number, Value};
use std::error::Error;
use std::fmt;
use std::str::FromStr;
use std::time::Duration;
use tokio::task::JoinHandle;
use tokio_postgres::types::{to_sql_checked, FromSql, IsNull, Kind, ToSql, Type};
use tokio_postgres::{Client, NoTls, Row};
use uuid::Uuid;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PostgresSslMode {
    #[default]
    Disable,
    Require,
}

impl FromStr for PostgresSslMode {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "disable" => Ok(Self::Disable),
            "require" => Ok(Self::Require),
            other => bail!("unsupported PostgreSQL sslmode `{other}`; expected disable or require"),
        }
    }
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PostgresConfig {
    pub server: String,
    #[serde(default = "default_port")]
    pub port: u16,
    pub database: String,
    pub user: String,
    pub password: Option<String>,
    #[serde(default)]
    pub sslmode: PostgresSslMode,
    #[serde(default = "default_request_timeout_ms")]
    pub request_timeout_ms: u64,
    #[serde(default = "default_max_rows")]
    pub max_rows: usize,
    #[serde(default)]
    pub allow_writes: bool,
    #[serde(default = "default_application_name")]
    pub application_name: String,
}

impl fmt::Debug for PostgresConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PostgresConfig")
            .field("server", &self.server)
            .field("port", &self.port)
            .field("database", &self.database)
            .field("user", &self.user)
            .field("password", &self.password.as_ref().map(|_| "[redacted]"))
            .field("sslmode", &self.sslmode)
            .field("request_timeout_ms", &self.request_timeout_ms)
            .field("max_rows", &self.max_rows)
            .field("allow_writes", &self.allow_writes)
            .field("application_name", &self.application_name)
            .finish()
    }
}

impl PostgresConfig {
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("server", self.server.as_str()),
            ("database", self.database.as_str()),
            ("user", self.user.as_str()),
            ("applicationName", self.application_name.as_str()),
        ] {
            if value.trim().is_empty() {
                bail!("{name} must not be empty");
            }
        }
        if self.max_rows == 0 || self.max_rows > 50_000 {
            bail!("maxRows must be between 1 and 50000");
        }
        if self.request_timeout_ms < 1_000 || self.request_timeout_ms > 300_000 {
            bail!("requestTimeoutMs must be between 1000 and 300000");
        }
        Ok(())
    }
}

pub struct PostgresAdapter {
    config: PostgresConfig,
    client: Option<Client>,
    connection_task: Option<JoinHandle<()>>,
    transaction_open: bool,
}

impl PostgresAdapter {
    pub fn new(config: PostgresConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            client: None,
            connection_task: None,
            transaction_open: false,
        })
    }

    pub fn config(&self) -> &PostgresConfig {
        &self.config
    }

    pub async fn query(&mut self, sql: &str, parameters: &[SqlParameter]) -> Result<Vec<Value>> {
        validate_read_statement(sql)?;
        self.query_inner(sql, parameters).await
    }

    pub async fn execute(&mut self, sql: &str, parameters: &[SqlParameter]) -> Result<u64> {
        self.ensure_writes()?;
        validate_write_statement(sql)?;
        let sql = map_parameter_placeholders(sql, parameters.len())?;
        let timeout = self.request_timeout();
        let client = self.client().await?;
        let postgres_parameters = postgres_parameters(parameters);
        let bound = bound_parameters(&postgres_parameters);
        tokio::time::timeout(timeout, client.execute(sql.as_str(), &bound))
            .await
            .context("PostgreSQL statement timed out")?
            .with_context(|| format!("PostgreSQL statement failed: {}", one_line(sql.as_str())))
    }

    pub async fn execute_returning(
        &mut self,
        sql: &str,
        parameters: &[SqlParameter],
    ) -> Result<Vec<Value>> {
        self.ensure_writes()?;
        validate_write_statement(sql)?;
        self.query_inner(sql, parameters).await
    }

    pub async fn begin_transaction(&mut self) -> Result<()> {
        self.ensure_writes()?;
        if self.transaction_open {
            bail!("nested SQL transactions are not supported");
        }
        let timeout = self.request_timeout();
        let client = self.client().await?;
        tokio::time::timeout(timeout, client.batch_execute("BEGIN"))
            .await
            .context("PostgreSQL begin transaction timed out")?
            .context("failed to begin PostgreSQL transaction")?;
        self.transaction_open = true;
        Ok(())
    }

    pub async fn commit_transaction(&mut self) -> Result<()> {
        if !self.transaction_open {
            bail!("no SQL transaction is open");
        }
        let timeout = self.request_timeout();
        let result = tokio::time::timeout(timeout, self.client().await?.batch_execute("COMMIT"))
            .await
            .context("PostgreSQL commit timed out")?
            .context("failed to commit PostgreSQL transaction");
        if result.is_ok() {
            self.transaction_open = false;
        }
        result
    }

    pub async fn rollback_transaction(&mut self) -> Result<()> {
        if !self.transaction_open {
            return Ok(());
        }
        let timeout = self.request_timeout();
        let result = tokio::time::timeout(timeout, self.client().await?.batch_execute("ROLLBACK"))
            .await
            .context("PostgreSQL rollback timed out")?
            .context("failed to roll back PostgreSQL transaction");
        self.transaction_open = false;
        result
    }

    async fn query_inner(&mut self, sql: &str, parameters: &[SqlParameter]) -> Result<Vec<Value>> {
        let sql = map_parameter_placeholders(sql, parameters.len())?;
        let max_rows = self.config.max_rows;
        let timeout = self.request_timeout();
        let client = self.client().await?;
        let postgres_parameters = postgres_parameters(parameters);
        let bound = bound_parameters(&postgres_parameters);
        let rows = tokio::time::timeout(timeout, client.query(sql.as_str(), &bound))
            .await
            .context("PostgreSQL query timed out")?
            .with_context(|| format!("PostgreSQL query failed: {}", one_line(sql.as_str())))?;
        enforce_max_rows(rows.len(), max_rows)?;
        rows.iter().map(row_to_json).collect()
    }

    async fn client(&mut self) -> Result<&Client> {
        if self.client.is_none() {
            let mut config = tokio_postgres::Config::new();
            config
                .host(&self.config.server)
                .port(self.config.port)
                .dbname(&self.config.database)
                .user(&self.config.user)
                .application_name(&self.config.application_name);
            if let Some(password) = self.config.password.as_deref() {
                config.password(password);
            }
            let timeout = self.request_timeout();
            let (client, connection_task) = match self.config.sslmode {
                PostgresSslMode::Disable => {
                    config.ssl_mode(tokio_postgres::config::SslMode::Disable);
                    let (client, connection) = tokio::time::timeout(timeout, config.connect(NoTls))
                        .await
                        .context("PostgreSQL connection timed out")?
                        .context("failed to connect to PostgreSQL")?;
                    let task = tokio::spawn(async move {
                        let _ = connection.await;
                    });
                    (client, task)
                }
                PostgresSslMode::Require => {
                    config.ssl_mode(tokio_postgres::config::SslMode::Require);
                    let connector = native_tls::TlsConnector::builder()
                        .build()
                        .context("failed to build PostgreSQL TLS connector")?;
                    let connector = MakeTlsConnector::new(connector);
                    let (client, connection) =
                        tokio::time::timeout(timeout, config.connect(connector))
                            .await
                            .context("PostgreSQL TLS connection timed out")?
                            .context("failed to connect to PostgreSQL with TLS")?;
                    let task = tokio::spawn(async move {
                        let _ = connection.await;
                    });
                    (client, task)
                }
            };
            self.client = Some(client);
            self.connection_task = Some(connection_task);
        }
        Ok(self.client.as_ref().expect("client initialized"))
    }

    fn request_timeout(&self) -> Duration {
        Duration::from_millis(self.config.request_timeout_ms)
    }

    fn ensure_writes(&self) -> Result<()> {
        if !self.config.allow_writes {
            bail!("SQL writes are disabled for this connection");
        }
        Ok(())
    }
}

impl Drop for PostgresAdapter {
    fn drop(&mut self) {
        if let Some(task) = self.connection_task.take() {
            task.abort();
        }
    }
}

#[derive(Debug)]
struct PostgresParameter<'a>(&'a SqlParameter);

fn postgres_parameters(parameters: &[SqlParameter]) -> Vec<PostgresParameter<'_>> {
    parameters.iter().map(PostgresParameter).collect()
}

fn bound_parameters<'a>(parameters: &'a [PostgresParameter<'a>]) -> Vec<&'a (dyn ToSql + Sync)> {
    parameters
        .iter()
        .map(|parameter| parameter as &(dyn ToSql + Sync))
        .collect()
}

impl ToSql for PostgresParameter<'_> {
    fn to_sql(
        &self,
        ty: &Type,
        out: &mut BytesMut,
    ) -> std::result::Result<IsNull, Box<dyn Error + Sync + Send>> {
        match self.0 {
            SqlParameter::NullString => Ok(IsNull::Yes),
            SqlParameter::String(value) => string_to_sql(value, ty, out),
            SqlParameter::Bytes(value) => value.as_slice().to_sql(ty, out),
            SqlParameter::Boolean(value) => value.to_sql(ty, out),
            SqlParameter::I16(value) => integer_to_sql(i64::from(*value), ty, out),
            SqlParameter::I32(value) => integer_to_sql(i64::from(*value), ty, out),
            SqlParameter::I64(value) => integer_to_sql(*value, ty, out),
            SqlParameter::F32(value) => float_to_sql(f64::from(*value), ty, out),
            SqlParameter::F64(value) => float_to_sql(*value, ty, out),
        }
    }

    fn accepts(_ty: &Type) -> bool {
        true
    }

    to_sql_checked!();
}

fn string_to_sql(
    value: &str,
    ty: &Type,
    out: &mut BytesMut,
) -> std::result::Result<IsNull, Box<dyn Error + Sync + Send>> {
    if <&str as ToSql>::accepts(ty) {
        return value.to_sql(ty, out);
    }
    match *ty {
        Type::UUID => Uuid::parse_str(value)?.to_sql(ty, out),
        Type::JSON => {
            out.extend_from_slice(value.as_bytes());
            Ok(IsNull::No)
        }
        Type::JSONB => {
            serde_json::from_str::<Value>(value)?;
            out.extend_from_slice(&[1]);
            out.extend_from_slice(value.as_bytes());
            Ok(IsNull::No)
        }
        _ if matches!(ty.kind(), Kind::Enum(_)) => {
            out.extend_from_slice(value.as_bytes());
            Ok(IsNull::No)
        }
        _ => Err(wrong_parameter_type("string", ty)),
    }
}

fn integer_to_sql(
    value: i64,
    ty: &Type,
    out: &mut BytesMut,
) -> std::result::Result<IsNull, Box<dyn Error + Sync + Send>> {
    match *ty {
        Type::INT2 => i16::try_from(value)?.to_sql(ty, out),
        Type::INT4 => i32::try_from(value)?.to_sql(ty, out),
        Type::INT8 => value.to_sql(ty, out),
        Type::OID => u32::try_from(value)?.to_sql(ty, out),
        _ => Err(wrong_parameter_type("integer", ty)),
    }
}

fn float_to_sql(
    value: f64,
    ty: &Type,
    out: &mut BytesMut,
) -> std::result::Result<IsNull, Box<dyn Error + Sync + Send>> {
    match *ty {
        Type::FLOAT4 => (value as f32).to_sql(ty, out),
        Type::FLOAT8 => value.to_sql(ty, out),
        _ => Err(wrong_parameter_type("floating-point", ty)),
    }
}

fn wrong_parameter_type(kind: &str, ty: &Type) -> Box<dyn Error + Sync + Send> {
    format!("cannot bind {kind} SQL parameter to PostgreSQL type {ty}").into()
}

fn row_to_json(row: &Row) -> Result<Value> {
    let mut object = Map::new();
    for (index, column) in row.columns().iter().enumerate() {
        let value = column_to_json(row, index, column.type_()).with_context(|| {
            format!(
                "failed to convert PostgreSQL column `{}` of type {}",
                column.name(),
                column.type_()
            )
        })?;
        object.insert(column.name().to_string(), value);
    }
    Ok(Value::Object(object))
}

fn column_to_json(row: &Row, index: usize, ty: &Type) -> Result<Value> {
    let value = match *ty {
        Type::BOOL => PostgresValue::Boolean(row.try_get(index)?),
        Type::INT2 => PostgresValue::I64(row.try_get::<_, Option<i16>>(index)?.map(i64::from)),
        Type::INT4 => PostgresValue::I64(row.try_get::<_, Option<i32>>(index)?.map(i64::from)),
        Type::INT8 => PostgresValue::I64(row.try_get(index)?),
        Type::OID => PostgresValue::U64(row.try_get::<_, Option<u32>>(index)?.map(u64::from)),
        Type::FLOAT4 => PostgresValue::F64(
            row.try_get::<_, Option<f32>>(index)?
                .map(|value| value as f64),
        ),
        Type::FLOAT8 => PostgresValue::F64(row.try_get(index)?),
        Type::TEXT | Type::VARCHAR | Type::BPCHAR | Type::NAME | Type::UNKNOWN => {
            PostgresValue::String(row.try_get(index)?)
        }
        Type::BYTEA => PostgresValue::Bytes(row.try_get(index)?),
        Type::UUID => PostgresValue::String(
            row.try_get::<_, Option<Uuid>>(index)?
                .map(|value| value.to_string()),
        ),
        Type::JSON | Type::JSONB => PostgresValue::Json(row.try_get(index)?),
        Type::TIMESTAMPTZ => PostgresValue::Timestamptz(row.try_get(index)?),
        Type::TIMESTAMP => PostgresValue::Timestamp(row.try_get(index)?),
        Type::DATE => PostgresValue::Date(row.try_get(index)?),
        Type::TIME => PostgresValue::Time(row.try_get(index)?),
        Type::NUMERIC => PostgresValue::String(
            row.try_get::<_, Option<PgNumeric>>(index)?
                .map(|value| value.0),
        ),
        _ if matches!(ty.kind(), Kind::Enum(_)) => PostgresValue::String(
            row.try_get::<_, Option<PgText>>(index)?
                .map(|value| value.0),
        ),
        _ => bail!("unsupported PostgreSQL result type {ty}"),
    };
    Ok(value.into_json())
}

#[derive(Debug)]
enum PostgresValue {
    Boolean(Option<bool>),
    I64(Option<i64>),
    U64(Option<u64>),
    F64(Option<f64>),
    String(Option<String>),
    Bytes(Option<Vec<u8>>),
    Json(Option<Value>),
    Timestamptz(Option<DateTime<Utc>>),
    Timestamp(Option<NaiveDateTime>),
    Date(Option<NaiveDate>),
    Time(Option<NaiveTime>),
}

impl PostgresValue {
    fn into_json(self) -> Value {
        match self {
            Self::Boolean(value) => value.map(Value::Bool).unwrap_or(Value::Null),
            Self::I64(value) => value.map(|value| json!(value)).unwrap_or(Value::Null),
            Self::U64(value) => value.map(|value| json!(value)).unwrap_or(Value::Null),
            Self::F64(value) => value
                .and_then(Number::from_f64)
                .map(Value::Number)
                .unwrap_or(Value::Null),
            Self::String(value) => value.map(Value::String).unwrap_or(Value::Null),
            Self::Bytes(value) => value
                .map(|value| Value::String(base64::engine::general_purpose::STANDARD.encode(value)))
                .unwrap_or(Value::Null),
            Self::Json(value) => value.unwrap_or(Value::Null),
            Self::Timestamptz(value) => value
                .map(|value| Value::String(value.to_rfc3339_opts(SecondsFormat::AutoSi, true)))
                .unwrap_or(Value::Null),
            Self::Timestamp(value) => value
                .map(|value| Value::String(value.format("%Y-%m-%dT%H:%M:%S%.f").to_string()))
                .unwrap_or(Value::Null),
            Self::Date(value) => value
                .map(|value| Value::String(value.to_string()))
                .unwrap_or(Value::Null),
            Self::Time(value) => value
                .map(|value| Value::String(value.format("%H:%M:%S%.f").to_string()))
                .unwrap_or(Value::Null),
        }
    }
}

struct PgText(String);

impl<'a> FromSql<'a> for PgText {
    fn from_sql(
        _ty: &Type,
        raw: &'a [u8],
    ) -> std::result::Result<Self, Box<dyn Error + Sync + Send>> {
        Ok(Self(std::str::from_utf8(raw)?.to_owned()))
    }

    fn accepts(_ty: &Type) -> bool {
        true
    }
}

struct PgNumeric(String);

impl<'a> FromSql<'a> for PgNumeric {
    fn from_sql(
        _ty: &Type,
        raw: &'a [u8],
    ) -> std::result::Result<Self, Box<dyn Error + Sync + Send>> {
        Ok(Self(decode_numeric(raw)?))
    }

    fn accepts(ty: &Type) -> bool {
        *ty == Type::NUMERIC
    }
}

fn decode_numeric(raw: &[u8]) -> std::result::Result<String, Box<dyn Error + Sync + Send>> {
    if raw.len() < 8 || (raw.len() - 8) % 2 != 0 {
        return Err("invalid PostgreSQL numeric payload".into());
    }
    let read_i16 = |offset: usize| i16::from_be_bytes([raw[offset], raw[offset + 1]]);
    let read_u16 = |offset: usize| u16::from_be_bytes([raw[offset], raw[offset + 1]]);
    let digits_len = usize::try_from(read_i16(0)).map_err(|_| "negative numeric digit count")?;
    if raw.len() != 8 + digits_len * 2 {
        return Err("invalid PostgreSQL numeric digit count".into());
    }
    let weight = i32::from(read_i16(2));
    let sign = read_u16(4);
    let scale = usize::from(read_u16(6));
    if scale > 100_000 || weight.unsigned_abs() > 100_000 {
        return Err("PostgreSQL numeric value is too large to convert".into());
    }
    match sign {
        0xC000 => return Ok("NaN".to_owned()),
        0xD000 => return Ok("Infinity".to_owned()),
        0xF000 => return Ok("-Infinity".to_owned()),
        0x0000 | 0x4000 => {}
        _ => return Err("invalid PostgreSQL numeric sign".into()),
    }
    let digits = (0..digits_len)
        .map(|index| read_u16(8 + index * 2))
        .collect::<Vec<_>>();
    if digits.iter().any(|digit| *digit >= 10_000) {
        return Err("invalid PostgreSQL numeric digit".into());
    }
    let digit_at_exponent = |exponent: i32| -> u16 {
        let index = weight - exponent;
        usize::try_from(index)
            .ok()
            .and_then(|index| digits.get(index))
            .copied()
            .unwrap_or(0)
    };
    let mut value = String::new();
    if weight >= 0 {
        value.push_str(&digit_at_exponent(weight).to_string());
        for exponent in (0..weight).rev() {
            value.push_str(&format!("{:04}", digit_at_exponent(exponent)));
        }
    } else {
        value.push('0');
    }
    if scale > 0 {
        value.push('.');
        let groups = scale.div_ceil(4);
        for group in 0..groups {
            value.push_str(&format!("{:04}", digit_at_exponent(-1 - group as i32)));
        }
        value.truncate(value.len() - (groups * 4 - scale));
    }
    let is_zero = value.bytes().all(|byte| matches!(byte, b'0' | b'.'));
    if sign == 0x4000 && !is_zero {
        value.insert(0, '-');
    }
    Ok(value)
}

fn enforce_max_rows(row_count: usize, max_rows: usize) -> Result<()> {
    if row_count > max_rows {
        bail!("query returned {row_count} rows, exceeding maxRows {max_rows}");
    }
    Ok(())
}

fn map_parameter_placeholders(sql: &str, parameter_count: usize) -> Result<String> {
    #[derive(Clone, Debug)]
    enum State {
        Normal,
        SingleQuote,
        DoubleQuote,
        LineComment,
        BlockComment(usize),
        DollarQuote(Vec<u8>),
    }

    fn dollar_quote_delimiter(bytes: &[u8], start: usize) -> Option<Vec<u8>> {
        if bytes.get(start) != Some(&b'$') {
            return None;
        }
        let mut end = start + 1;
        if bytes.get(end) == Some(&b'$') {
            return Some(b"$$".to_vec());
        }
        let first = *bytes.get(end)?;
        if !(first.is_ascii_alphabetic() || first == b'_') {
            return None;
        }
        end += 1;
        while bytes
            .get(end)
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
        {
            end += 1;
        }
        (bytes.get(end) == Some(&b'$')).then(|| bytes[start..=end].to_vec())
    }

    let bytes = sql.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut used = vec![false; parameter_count];
    let mut state = State::Normal;
    let mut index = 0;
    while index < bytes.len() {
        match &mut state {
            State::Normal if bytes[index] == b'\'' => {
                output.push(bytes[index]);
                state = State::SingleQuote;
                index += 1;
            }
            State::Normal if bytes[index] == b'"' => {
                output.push(bytes[index]);
                state = State::DoubleQuote;
                index += 1;
            }
            State::Normal if bytes[index..].starts_with(b"--") => {
                output.extend_from_slice(b"--");
                state = State::LineComment;
                index += 2;
            }
            State::Normal if bytes[index..].starts_with(b"/*") => {
                output.extend_from_slice(b"/*");
                state = State::BlockComment(1);
                index += 2;
            }
            State::Normal if bytes[index] == b'$' => {
                if let Some(delimiter) = dollar_quote_delimiter(bytes, index) {
                    output.extend_from_slice(&delimiter);
                    index += delimiter.len();
                    state = State::DollarQuote(delimiter);
                } else {
                    output.push(bytes[index]);
                    index += 1;
                }
            }
            State::Normal
                if bytes[index] == b'@'
                    && bytes
                        .get(index + 1)
                        .is_some_and(|byte| matches!(byte, b'P' | b'p'))
                    && bytes
                        .get(index + 2)
                        .is_some_and(|byte| byte.is_ascii_digit())
                    && (index == 0
                        || !bytes[index - 1].is_ascii_alphanumeric()
                            && bytes[index - 1] != b'_') =>
            {
                let mut end = index + 2;
                while bytes.get(end).is_some_and(u8::is_ascii_digit) {
                    end += 1;
                }
                if bytes
                    .get(end)
                    .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
                {
                    output.extend_from_slice(&bytes[index..end]);
                    index = end;
                    continue;
                }
                let number = std::str::from_utf8(&bytes[index + 2..end])?
                    .parse::<usize>()
                    .context("invalid SQL parameter placeholder")?;
                if number == 0 || number > parameter_count {
                    bail!(
                        "SQL parameter placeholder @P{number} has no matching parameter (received {parameter_count})"
                    );
                }
                used[number - 1] = true;
                output.extend_from_slice(format!("${number}").as_bytes());
                index = end;
            }
            State::Normal => {
                output.push(bytes[index]);
                index += 1;
            }
            State::SingleQuote => {
                output.push(bytes[index]);
                if bytes[index] == b'\'' {
                    if bytes.get(index + 1) == Some(&b'\'') {
                        output.push(b'\'');
                        index += 2;
                    } else {
                        state = State::Normal;
                        index += 1;
                    }
                } else {
                    index += 1;
                }
            }
            State::DoubleQuote => {
                output.push(bytes[index]);
                if bytes[index] == b'"' {
                    if bytes.get(index + 1) == Some(&b'"') {
                        output.push(b'"');
                        index += 2;
                    } else {
                        state = State::Normal;
                        index += 1;
                    }
                } else {
                    index += 1;
                }
            }
            State::LineComment => {
                output.push(bytes[index]);
                if bytes[index] == b'\n' {
                    state = State::Normal;
                }
                index += 1;
            }
            State::BlockComment(depth) => {
                if bytes[index..].starts_with(b"/*") {
                    output.extend_from_slice(b"/*");
                    *depth += 1;
                    index += 2;
                } else if bytes[index..].starts_with(b"*/") {
                    output.extend_from_slice(b"*/");
                    *depth -= 1;
                    index += 2;
                    if *depth == 0 {
                        state = State::Normal;
                    }
                } else {
                    output.push(bytes[index]);
                    index += 1;
                }
            }
            State::DollarQuote(delimiter) => {
                if bytes[index..].starts_with(delimiter) {
                    output.extend_from_slice(delimiter);
                    index += delimiter.len();
                    state = State::Normal;
                } else {
                    output.push(bytes[index]);
                    index += 1;
                }
            }
        }
    }
    if let Some(missing) = used.iter().position(|used| !used) {
        bail!(
            "SQL does not reference parameter @P{} (received {parameter_count} parameters)",
            missing + 1
        );
    }
    String::from_utf8(output).context("rewritten SQL was not valid UTF-8")
}

fn default_port() -> u16 {
    5432
}

fn default_request_timeout_ms() -> u64 {
    30_000
}

fn default_max_rows() -> usize {
    5_000
}

fn default_application_name() -> String {
    "ctox-external-sql-sync".to_owned()
}

fn one_line(value: &str) -> String {
    let value = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if value.len() > 240 {
        format!("{}...", &value[..240])
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn config(allow_writes: bool) -> PostgresConfig {
        PostgresConfig {
            server: "127.0.0.1".into(),
            port: 5432,
            database: "ctox".into(),
            user: "ctox".into(),
            password: Some("secret".into()),
            sslmode: PostgresSslMode::Disable,
            request_timeout_ms: 30_000,
            max_rows: 5_000,
            allow_writes,
            application_name: default_application_name(),
        }
    }

    #[test]
    fn config_defaults_are_postgres_safe_and_passwords_are_redacted() {
        let config: PostgresConfig = serde_json::from_value(json!({
            "server": "127.0.0.1",
            "database": "eventus",
            "user": "ctox",
            "password": "secret"
        }))
        .expect("config");
        config.validate().expect("valid config");
        assert_eq!(config.port, 5432);
        assert_eq!(config.sslmode, PostgresSslMode::Disable);
        assert_eq!(config.request_timeout_ms, 30_000);
        assert_eq!(config.max_rows, 5_000);
        assert!(!config.allow_writes);
        let debug = format!("{config:?}");
        assert!(debug.contains("[redacted]"));
        assert!(!debug.contains("secret"));
    }

    #[test]
    fn maps_sqlserver_style_parameters_without_touching_literals_or_comments() {
        let sql = "SELECT @P1, '@P2', \"@P2\", $$@P2$$, @p2 -- @P1\n/* @P2 */";
        assert_eq!(
            map_parameter_placeholders(sql, 2).expect("mapped SQL"),
            "SELECT $1, '@P2', \"@P2\", $$@P2$$, $2 -- @P1\n/* @P2 */"
        );
        assert!(map_parameter_placeholders("SELECT @P2", 2).is_err());
        assert!(map_parameter_placeholders("SELECT @P3", 2).is_err());
    }

    #[test]
    fn max_rows_is_enforced_at_the_configured_boundary() {
        enforce_max_rows(2, 2).expect("boundary allowed");
        let error = enforce_max_rows(3, 2).expect_err("too many rows");
        assert!(error.to_string().contains("exceeding maxRows 2"));
    }

    #[test]
    fn writes_are_fail_closed() {
        let adapter = PostgresAdapter::new(config(false)).expect("adapter");
        assert!(adapter.ensure_writes().is_err());
        let adapter = PostgresAdapter::new(config(true)).expect("adapter");
        adapter.ensure_writes().expect("writes enabled");
    }

    #[test]
    fn values_preserve_null_and_render_timestamptz_as_iso_8601() {
        assert_eq!(PostgresValue::String(None).into_json(), Value::Null);
        let timestamp = Utc
            .with_ymd_and_hms(2026, 7, 28, 10, 11, 12)
            .single()
            .expect("timestamp");
        assert_eq!(
            PostgresValue::Timestamptz(Some(timestamp)).into_json(),
            json!("2026-07-28T10:11:12Z")
        );
    }

    #[test]
    fn numeric_binary_values_follow_sqlserver_string_convention() {
        let raw = [
            0, 2, // ndigits
            0, 0, // weight
            0, 0, // positive
            0, 2, // scale
            0, 12, // 12
            13, 128, // 3456
        ];
        assert_eq!(decode_numeric(&raw).expect("numeric"), "12.34");
    }
}
