use anyhow::{anyhow, bail, Context, Result};
use candle_core::{Device, Tensor};
use half::bf16;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TorchStorageKind {
    BFloat16,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TorchTensorArchive {
    pub storage_kind: TorchStorageKind,
    pub storage_key: String,
    pub device: String,
    pub numel: usize,
    pub storage_offset: usize,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    pub data: Vec<u8>,
}

impl TorchTensorArchive {
    pub fn into_tensor(self, device: &Device) -> Result<Tensor> {
        match self.storage_kind {
            TorchStorageKind::BFloat16 => {
                let expected_len = self
                    .numel
                    .checked_mul(2)
                    .ok_or_else(|| anyhow!("voice embedding byte length overflow"))?;
                if self.data.len() != expected_len {
                    bail!(
                        "voice embedding payload length mismatch: got {}, expected {}",
                        self.data.len(),
                        expected_len
                    );
                }
                let values = self
                    .data
                    .chunks_exact(2)
                    .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
                    .collect::<Vec<_>>();
                let tensor = Tensor::from_vec(values, self.shape.clone(), device)?;
                Ok(tensor)
            }
        }
    }
}

pub fn load_voice_embedding_pt(path: impl AsRef<Path>) -> Result<TorchTensorArchive> {
    let raw = fs::read(path.as_ref())
        .with_context(|| format!("failed to read voice embedding {}", path.as_ref().display()))?;
    let zip_entries = parse_stored_zip_entries(&raw)?;
    let pkl = zip_entries
        .iter()
        .find_map(|(name, data)| name.ends_with("data.pkl").then_some(data.as_slice()))
        .ok_or_else(|| anyhow!("voice embedding archive missing data.pkl"))?;
    let data = zip_entries
        .iter()
        .find_map(|(name, data)| name.ends_with("data/0").then_some(data.clone()))
        .ok_or_else(|| anyhow!("voice embedding archive missing tensor storage data"))?;

    let mut tensor = parse_simple_torch_tensor_pickle(pkl)?;
    tensor.data = data;
    Ok(tensor)
}

fn parse_stored_zip_entries(raw: &[u8]) -> Result<BTreeMap<String, Vec<u8>>> {
    let eocd_offset = raw
        .windows(4)
        .rposition(|window| window == b"PK\x05\x06")
        .ok_or_else(|| anyhow!("voice embedding archive missing EOCD"))?;
    let central_dir_size = read_u32_le(raw, eocd_offset + 12)? as usize;
    let central_dir_offset = read_u32_le(raw, eocd_offset + 16)? as usize;
    let mut cursor = central_dir_offset;
    let end = central_dir_offset + central_dir_size;
    let mut entries = BTreeMap::new();

    while cursor < end {
        ensure_sig(raw, cursor, b"PK\x01\x02")?;
        let compression_method = read_u16_le(raw, cursor + 10)?;
        let compressed_size = read_u32_le(raw, cursor + 20)? as usize;
        let uncompressed_size = read_u32_le(raw, cursor + 24)? as usize;
        let file_name_len = read_u16_le(raw, cursor + 28)? as usize;
        let extra_len = read_u16_le(raw, cursor + 30)? as usize;
        let comment_len = read_u16_le(raw, cursor + 32)? as usize;
        let local_header_offset = read_u32_le(raw, cursor + 42)? as usize;
        let name_start = cursor + 46;
        let name_end = name_start + file_name_len;
        let name = std::str::from_utf8(get_slice(raw, name_start, name_end)?)
            .context("invalid UTF-8 zip entry name")?
            .to_string();
        if compression_method != 0 {
            bail!(
                "voice embedding archive entry `{name}` uses compression method {compression_method}, but only stored entries are supported"
            );
        }
        let data = extract_stored_local_file(raw, local_header_offset, compressed_size)?;
        if data.len() != uncompressed_size {
            bail!(
                "voice embedding archive entry `{name}` size mismatch: got {}, expected {}",
                data.len(),
                uncompressed_size
            );
        }
        entries.insert(name, data);
        cursor = name_end + extra_len + comment_len;
    }

    Ok(entries)
}

fn extract_stored_local_file(raw: &[u8], offset: usize, size: usize) -> Result<Vec<u8>> {
    ensure_sig(raw, offset, b"PK\x03\x04")?;
    let file_name_len = read_u16_le(raw, offset + 26)? as usize;
    let extra_len = read_u16_le(raw, offset + 28)? as usize;
    let data_start = offset + 30 + file_name_len + extra_len;
    let data_end = data_start + size;
    Ok(get_slice(raw, data_start, data_end)?.to_vec())
}

fn parse_simple_torch_tensor_pickle(raw: &[u8]) -> Result<TorchTensorArchive> {
    let mut cursor = 0usize;
    expect_byte(raw, &mut cursor, 0x80)?;
    let _proto = next_byte(raw, &mut cursor)?;
    expect_byte(raw, &mut cursor, b'c')?;
    let global = read_global(raw, &mut cursor)?;
    if global != "torch._utils _rebuild_tensor_v2" {
        bail!("unsupported torch tensor pickle root `{global}`");
    }
    skip_binput(raw, &mut cursor)?;
    expect_byte(raw, &mut cursor, b'(')?;
    expect_byte(raw, &mut cursor, b'(')?;
    let storage_word = read_binunicode(raw, &mut cursor)?;
    if storage_word != "storage" {
        bail!("unexpected torch pickle storage marker `{storage_word}`");
    }
    skip_binput(raw, &mut cursor)?;
    expect_byte(raw, &mut cursor, b'c')?;
    let storage_global = read_global(raw, &mut cursor)?;
    let storage_kind = match storage_global.as_str() {
        "torch BFloat16Storage" => TorchStorageKind::BFloat16,
        other => bail!("unsupported voice embedding storage kind `{other}`"),
    };
    skip_binput(raw, &mut cursor)?;
    let storage_key = read_binunicode(raw, &mut cursor)?;
    skip_binput(raw, &mut cursor)?;
    let device = read_binunicode(raw, &mut cursor)?;
    skip_binput(raw, &mut cursor)?;
    let numel = read_pickle_int(raw, &mut cursor)? as usize;
    expect_byte(raw, &mut cursor, b't')?;
    skip_binput(raw, &mut cursor)?;
    expect_byte(raw, &mut cursor, b'Q')?;
    let storage_offset = read_pickle_int(raw, &mut cursor)? as usize;
    let shape = read_pickle_tuple_usize(raw, &mut cursor)?;
    skip_binput(raw, &mut cursor)?;
    let stride = read_pickle_tuple_usize(raw, &mut cursor)?;

    Ok(TorchTensorArchive {
        storage_kind,
        storage_key,
        device,
        numel,
        storage_offset,
        shape,
        stride,
        data: Vec::new(),
    })
}

fn skip_binput(raw: &[u8], cursor: &mut usize) -> Result<()> {
    if peek_byte(raw, *cursor)? == b'q' {
        *cursor += 2;
    }
    Ok(())
}

fn read_pickle_tuple_usize(raw: &[u8], cursor: &mut usize) -> Result<Vec<usize>> {
    let opcode = next_byte(raw, cursor)?;
    match opcode {
        b')' => Ok(Vec::new()),
        b'K' | b'M' | b'J' => read_compact_pickle_tuple(raw, cursor, opcode),
        b'(' => {
            let mut values = Vec::new();
            loop {
                let peek = peek_byte(raw, *cursor)?;
                if peek == b't' {
                    *cursor += 1;
                    break;
                }
                values.push(read_pickle_int(raw, cursor)? as usize);
            }
            Ok(values)
        }
        0x85 => Ok(Vec::new()),
        0x86 => bail!("unsupported tuple2 opcode without preceding values"),
        0x87 => bail!("unsupported tuple3 opcode without preceding values"),
        other => bail!("unsupported tuple opcode in voice embedding pickle: {other:#x}"),
    }
}

fn read_compact_pickle_tuple(
    raw: &[u8],
    cursor: &mut usize,
    first_opcode: u8,
) -> Result<Vec<usize>> {
    let mut values = vec![read_pickle_int_from_opcode(raw, cursor, first_opcode)? as usize];
    while is_pickle_int_opcode(peek_byte(raw, *cursor)?) {
        values.push(read_pickle_int(raw, cursor)? as usize);
    }
    match peek_byte(raw, *cursor)? {
        0x85 if values.len() == 1 => {
            *cursor += 1;
            Ok(values)
        }
        0x86 if values.len() == 2 => {
            *cursor += 1;
            Ok(values)
        }
        0x87 if values.len() == 3 => {
            *cursor += 1;
            Ok(values)
        }
        _ if values.len() == 1 => Ok(values),
        other => bail!(
            "unsupported compact tuple in voice embedding pickle: values={values:?}, tuple_opcode={other:#x}"
        ),
    }
}

fn is_pickle_int_opcode(opcode: u8) -> bool {
    matches!(opcode, b'K' | b'M' | b'J')
}

fn read_pickle_int_from_opcode(raw: &[u8], cursor: &mut usize, opcode: u8) -> Result<i64> {
    match opcode {
        b'K' => Ok(next_byte(raw, cursor)? as i64),
        b'M' => {
            let lo = next_byte(raw, cursor)?;
            let hi = next_byte(raw, cursor)?;
            Ok(u16::from_le_bytes([lo, hi]) as i64)
        }
        b'J' => {
            let bytes = [
                next_byte(raw, cursor)?,
                next_byte(raw, cursor)?,
                next_byte(raw, cursor)?,
                next_byte(raw, cursor)?,
            ];
            Ok(i32::from_le_bytes(bytes) as i64)
        }
        other => bail!("unsupported pickle integer opcode {other:#x}"),
    }
}

fn read_pickle_int(raw: &[u8], cursor: &mut usize) -> Result<i64> {
    let opcode = next_byte(raw, cursor)?;
    read_pickle_int_from_opcode(raw, cursor, opcode)
}

fn read_binunicode(raw: &[u8], cursor: &mut usize) -> Result<String> {
    expect_byte(raw, cursor, b'X')?;
    let len = read_u32_le(raw, *cursor)? as usize;
    *cursor += 4;
    let end = *cursor + len;
    let s = std::str::from_utf8(get_slice(raw, *cursor, end)?)
        .context("invalid UTF-8 in torch pickle string")?
        .to_string();
    *cursor = end;
    Ok(s)
}

fn read_global(raw: &[u8], cursor: &mut usize) -> Result<String> {
    let start = *cursor;
    let nl1 = raw[start..]
        .iter()
        .position(|b| *b == b'\n')
        .ok_or_else(|| anyhow!("unterminated pickle GLOBAL module"))?
        + start;
    let nl2 = raw[nl1 + 1..]
        .iter()
        .position(|b| *b == b'\n')
        .ok_or_else(|| anyhow!("unterminated pickle GLOBAL name"))?
        + nl1
        + 1;
    let module = std::str::from_utf8(&raw[start..nl1]).context("invalid pickle GLOBAL module")?;
    let name = std::str::from_utf8(&raw[nl1 + 1..nl2]).context("invalid pickle GLOBAL name")?;
    *cursor = nl2 + 1;
    Ok(format!("{module} {name}"))
}

fn ensure_sig(raw: &[u8], offset: usize, sig: &[u8; 4]) -> Result<()> {
    if get_slice(raw, offset, offset + 4)? != sig {
        bail!("unexpected zip signature at offset {offset}");
    }
    Ok(())
}

fn read_u16_le(raw: &[u8], offset: usize) -> Result<u16> {
    Ok(u16::from_le_bytes(
        get_slice(raw, offset, offset + 2)?
            .try_into()
            .expect("slice length checked"),
    ))
}

fn read_u32_le(raw: &[u8], offset: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(
        get_slice(raw, offset, offset + 4)?
            .try_into()
            .expect("slice length checked"),
    ))
}

fn get_slice(raw: &[u8], start: usize, end: usize) -> Result<&[u8]> {
    raw.get(start..end)
        .ok_or_else(|| anyhow!("out-of-bounds voice embedding archive access {start}..{end}"))
}

fn expect_byte(raw: &[u8], cursor: &mut usize, expected: u8) -> Result<()> {
    let got = next_byte(raw, cursor)?;
    if got != expected {
        bail!(
            "unexpected pickle opcode {got:#x} at byte {}, expected {expected:#x}",
            cursor.saturating_sub(1)
        );
    }
    Ok(())
}

fn next_byte(raw: &[u8], cursor: &mut usize) -> Result<u8> {
    let byte = *raw
        .get(*cursor)
        .ok_or_else(|| anyhow!("unexpected EOF while parsing voice embedding archive"))?;
    *cursor += 1;
    Ok(byte)
}

fn peek_byte(raw: &[u8], cursor: usize) -> Result<u8> {
    raw.get(cursor)
        .copied()
        .ok_or_else(|| anyhow!("unexpected EOF while peeking voice embedding archive"))
}

#[cfg(test)]
mod tests {
    use super::{
        load_voice_embedding_pt, parse_simple_torch_tensor_pickle, parse_stored_zip_entries,
    };
    use std::ffi::OsString;
    use std::io::Write;

    fn synth_pickle() -> Vec<u8> {
        vec![
            0x80, 0x02, b'c', b't', b'o', b'r', b'c', b'h', b'.', b'_', b'u', b't', b'i', b'l',
            b's', b'\n', b'_', b'r', b'e', b'b', b'u', b'i', b'l', b'd', b'_', b't', b'e', b'n',
            b's', b'o', b'r', b'_', b'v', b'2', b'\n', b'q', 0x00, b'(', b'(', b'X', 0x07, 0x00,
            0x00, 0x00, b's', b't', b'o', b'r', b'a', b'g', b'e', b'q', 0x01, b'c', b't', b'o',
            b'r', b'c', b'h', b'\n', b'B', b'F', b'l', b'o', b'a', b't', b'1', b'6', b'S', b't',
            b'o', b'r', b'a', b'g', b'e', b'\n', b'q', 0x02, b'X', 0x01, 0x00, 0x00, 0x00, b'0',
            b'q', 0x03, b'X', 0x05, 0x00, 0x00, 0x00, b'c', b'p', b'u', b':', b'0', b'q', 0x04,
            b'J', 0x08, 0x00, 0x00, 0x00, b't', b'q', 0x05, b'Q', b'K', 0x00, b'(', b'K', 0x02,
            b'K', 0x04, b't', b'q', 0x06, b'(', b'K', 0x04, b'K', 0x01, b't', b'q', 0x07, 0x89,
            b'c', b'c', b'o', b'l', b'l', b'e', b'c', b't', b'i', b'o', b'n', b's', b'\n', b'O',
            b'r', b'd', b'e', b'r', b'e', b'd', b'D', b'i', b'c', b't', b'\n', b'q', 0x08, b')',
            b'R', b'q', 0x09, b't', b'q', 0x0A, b'R', b'q', 0x0B, b'.',
        ]
    }

    fn synth_compact_tuple_pickle() -> Vec<u8> {
        vec![
            0x80, 0x02, b'c', b't', b'o', b'r', b'c', b'h', b'.', b'_', b'u', b't', b'i', b'l',
            b's', b'\n', b'_', b'r', b'e', b'b', b'u', b'i', b'l', b'd', b'_', b't', b'e', b'n',
            b's', b'o', b'r', b'_', b'v', b'2', b'\n', b'q', 0x00, b'(', b'(', b'X', 0x07, 0x00,
            0x00, 0x00, b's', b't', b'o', b'r', b'a', b'g', b'e', b'q', 0x01, b'c', b't', b'o',
            b'r', b'c', b'h', b'\n', b'B', b'F', b'l', b'o', b'a', b't', b'1', b'6', b'S', b't',
            b'o', b'r', b'a', b'g', b'e', b'\n', b'q', 0x02, b'X', 0x01, 0x00, 0x00, 0x00, b'0',
            b'q', 0x03, b'X', 0x06, 0x00, 0x00, 0x00, b'c', b'u', b'd', b'a', b':', b'0', b'q',
            0x04, b'J', 0x00, 0x24, 0x03, 0x00, b't', b'q', 0x05, b'Q', b'K', 0x00, b'K', 0x43,
            b'M', 0x00, 0x0c, 0x86, b'q', 0x06, b'M', 0x00, 0x0c, b'K', 0x01, 0x86, b'q', 0x07,
            0x89, b'c', b'c', b'o', b'l', b'l', b'e', b'c', b't', b'i', b'o', b'n', b's', b'\n',
            b'O', b'r', b'd', b'e', b'r', b'e', b'd', b'D', b'i', b'c', b't', b'\n', b'q', 0x08,
            b')', b'R', b'q', 0x09, b't', b'q', 0x0A, b'R', b'q', 0x0B, b'.',
        ]
    }

    fn synth_stored_zip() -> Vec<u8> {
        let files = vec![
            ("voice_embed/data.pkl".to_string(), synth_pickle()),
            ("voice_embed/data/0".to_string(), vec![0u8; 16]),
        ];
        let mut out = Vec::new();
        let mut central = Vec::new();
        let mut offsets = Vec::new();

        for (name, data) in &files {
            let offset = out.len() as u32;
            offsets.push(offset);
            out.extend_from_slice(b"PK\x03\x04");
            out.extend_from_slice(&20u16.to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(data);
        }

        let central_offset = out.len() as u32;
        for ((name, data), offset) in files.iter().zip(offsets.iter()) {
            central.extend_from_slice(b"PK\x01\x02");
            central.extend_from_slice(&20u16.to_le_bytes());
            central.extend_from_slice(&20u16.to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&0u32.to_le_bytes());
            central.extend_from_slice(&(data.len() as u32).to_le_bytes());
            central.extend_from_slice(&(data.len() as u32).to_le_bytes());
            central.extend_from_slice(&(name.len() as u16).to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&0u16.to_le_bytes());
            central.extend_from_slice(&0u32.to_le_bytes());
            central.extend_from_slice(&offset.to_le_bytes());
            central.extend_from_slice(name.as_bytes());
        }
        out.extend_from_slice(&central);
        out.extend_from_slice(b"PK\x05\x06");
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&(files.len() as u16).to_le_bytes());
        out.extend_from_slice(&(files.len() as u16).to_le_bytes());
        out.extend_from_slice(&(central.len() as u32).to_le_bytes());
        out.extend_from_slice(&central_offset.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out
    }

    #[test]
    fn parses_simple_tensor_pickle() {
        let tensor = parse_simple_torch_tensor_pickle(&synth_pickle()).unwrap();
        assert_eq!(tensor.numel, 8);
        assert_eq!(tensor.shape, vec![2, 4]);
        assert_eq!(tensor.stride, vec![4, 1]);
        assert_eq!(tensor.device, "cpu:0");
    }

    #[test]
    fn parses_compact_tuple_tensor_pickle() {
        let tensor = parse_simple_torch_tensor_pickle(&synth_compact_tuple_pickle()).unwrap();
        assert_eq!(tensor.numel, 205824);
        assert_eq!(tensor.shape, vec![67, 3072]);
        assert_eq!(tensor.stride, vec![3072, 1]);
        assert_eq!(tensor.device, "cuda:0");
    }

    #[test]
    fn parses_stored_zip_entries() {
        let entries = parse_stored_zip_entries(&synth_stored_zip()).unwrap();
        assert!(entries.contains_key("voice_embed/data.pkl"));
        assert!(entries.contains_key("voice_embed/data/0"));
    }

    #[test]
    fn loads_voice_embedding_archive() {
        let mut path = std::env::temp_dir();
        let mut name = OsString::from("voxtral_voice_embedding_test_");
        name.push(std::process::id().to_string());
        name.push("_voice.pt");
        path.push(name);
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&synth_stored_zip()).unwrap();
        drop(file);
        let tensor = load_voice_embedding_pt(&path).unwrap();
        assert_eq!(tensor.shape, vec![2, 4]);
        assert_eq!(tensor.data.len(), 16);
        std::fs::remove_file(path).unwrap();
    }
}
