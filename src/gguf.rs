//! GGUF file format reader.
//!
//! This module provides functionality to read vocabulary and tokenizer metadata from
//! GGUF (GPT-Generated Unified Format) model files. GGUF is the standard format used
//! by llama.cpp and compatible inference engines.
//!
//! # Supported Versions
//!
//! - GGUF v2: Standard format
//! - GGUF v3: Extended format (used by GPT-2 and newer models)
//!
//! # Security
//!
//! This parser includes protections against malicious files:
//! - String allocation limits to prevent OOM attacks
//! - Bounds checking on array sizes
//! - Validation of file structure
//!
//! # Reference
//!
//! - [GGUF Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
//! - [llama.cpp GGUF support](https://github.com/ggerganov/llama.cpp)

use crate::{Error, TokenType};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Special token IDs loaded from GGUF metadata.
///
/// Groups the full set of llama.cpp special tokens so they can be passed
/// around as a single value rather than 8+ individual `Option<u32>`s.
/// Field names mirror the GGUF key suffixes exactly for traceability.
#[derive(Debug, Default, Clone, Copy)]
pub struct SpecialTokenIds {
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub unk: Option<u32>,
    pub pad: Option<u32>,
    pub eot: Option<u32>,
    pub eog: Option<u32>,
    pub sep: Option<u32>,
    pub nl: Option<u32>,
    pub fim_pre: Option<u32>,
    pub fim_suf: Option<u32>,
    pub fim_mid: Option<u32>,
    pub mask: Option<u32>,
}

/// Flags that control tokenization and normalization behaviour.
#[derive(Debug, Clone, Copy)]
pub struct TokenizationFlags {
    pub add_bos_token: bool,
    pub add_eos_token: bool,
    pub add_space_prefix: bool,
    pub clean_spaces: bool,
    pub remove_extra_whitespaces: bool,
    pub escape_whitespaces: bool,
    pub treat_whitespace_as_suffix: bool,
}

impl Default for TokenizationFlags {
    fn default() -> Self {
        Self {
            add_bos_token: true,
            add_eos_token: false,
            add_space_prefix: true,
            clean_spaces: false,
            remove_extra_whitespaces: false,
            escape_whitespaces: false,
            treat_whitespace_as_suffix: false,
        }
    }
}

/// Metadata extracted from a GGUF file's tokenizer section.
///
/// Contains the vocabulary, merge rules, and configuration needed to
/// reconstruct a tokenizer that matches the model's training.
pub struct GGUFMetadata {
    pub tokens: Vec<String>,
    pub scores: Option<Vec<f32>>,
    pub token_types: Option<Vec<TokenType>>,
    pub model_type: String,
    pub pre_type: Option<String>,
    /// Raw Jinja2 chat template string embedded in the GGUF file.
    ///
    /// This is the `tokenizer.chat_template` field from the model's metadata.
    /// Pass it to a Jinja renderer (e.g. [`shimmyjinja`]) together with your
    /// messages to produce a correctly formatted prompt string.
    ///
    /// [`shimmyjinja`]: https://crates.io/crates/shimmyjinja
    pub chat_template: Option<String>,
    pub special: SpecialTokenIds,
    pub flags: TokenizationFlags,
    pub merges: Option<Vec<(String, String)>>,
}

/// Loads tokenizer metadata from a GGUF file at the given path.
///
/// Reads only the metadata section; tensor data is skipped entirely. This
/// makes loading cheap even for multi-gigabyte model files.
///
/// # Errors
///
/// Returns [`Error::Io`] for file-system problems, [`Error::InvalidMetadata`]
/// for malformed or unsupported GGUF data, and [`Error::VocabularyError`] if
/// the tokenizer section is missing or inconsistent.
pub fn load_metadata<P: AsRef<Path>>(path: P) -> Result<GGUFMetadata, Error> {
    let file = File::open(path)?;
    load_metadata_from_reader(BufReader::new(file))
}

/// Loads tokenizer metadata from any [`Read`] source.
///
/// Identical to [`load_metadata`] but accepts an arbitrary reader, enabling
/// in-memory loading (e.g. from a `Cursor<&[u8]>` or a network stream).
///
/// # Errors
///
/// Same as [`load_metadata`].
pub fn load_metadata_from_reader<R: Read>(mut reader: R) -> Result<GGUFMetadata, Error> {
    /// Extract an optional `u32` from a kv-pair map.
    macro_rules! kv_u32 {
        ($map:expr, $key:expr) => {
            match $map.get($key) {
                Some(Value::U32(v)) => Some(*v),
                _ => None,
            }
        };
    }

    /// Extract an optional `bool` from a kv-pair map.
    macro_rules! kv_bool {
        ($map:expr, $key:expr) => {
            match $map.get($key) {
                Some(Value::Bool(v)) => Some(*v),
                _ => None,
            }
        };
    }

    // Track total string allocation to prevent OOM (Issue R2#3)
    let mut total_string_bytes: usize = 0;

    // Read magic
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;

    if &magic != b"GGUF" {
        return Err(Error::InvalidMetadata("Not a GGUF file".into()));
    }

    // Read version
    let version = read_u32(&mut reader)?;
    // Support both version 2 and 3 (GPT-2 uses v3)
    if !(2..=3).contains(&version) {
        return Err(Error::InvalidMetadata(format!(
            "Unsupported GGUF version: {version} (only versions 2-3 are supported)"
        )));
    }

    // Read counts
    let _tensor_count = read_u64(&mut reader)?;
    let metadata_count = read_u64(&mut reader)?;

    // Read metadata key-value pairs
    let mut kv_pairs = HashMap::new();
    for _ in 0..metadata_count {
        let key = read_string(&mut reader, &mut total_string_bytes)?;
        let value = read_value(&mut reader, &mut total_string_bytes)?;
        kv_pairs.insert(key, value);
    }

    // Extract tokenizer metadata
    let tokens = match kv_pairs.get("tokenizer.ggml.tokens") {
        Some(Value::StringArray(arr)) => arr.clone(),
        _ => {
            return Err(Error::InvalidMetadata(
                "Missing tokenizer.ggml.tokens".into(),
            ))
        }
    };

    let scores = match kv_pairs.get("tokenizer.ggml.scores") {
        Some(Value::F32Array(arr)) => Some(arr.clone()),
        _ => None,
    };

    let token_types = match kv_pairs.get("tokenizer.ggml.token_type") {
        Some(Value::I32Array(arr)) => Some(arr.iter().map(|&t| TokenType::from(t)).collect()),
        _ => None,
    };

    let model_type = match kv_pairs.get("tokenizer.ggml.model") {
        Some(Value::String(s)) => s.clone(),
        _ => "llama".to_string(), // Default
    };

    let pre_type = match kv_pairs.get("tokenizer.ggml.pre") {
        Some(Value::String(s)) => Some(s.clone()),
        _ => None,
    };

    // Chat template — raw Jinja2 string, pass to shimmyjinja to render prompts
    let chat_template = match kv_pairs.get("tokenizer.chat_template") {
        Some(Value::String(s)) => Some(s.clone()),
        _ => None,
    };

    // Special tokens
    let special = SpecialTokenIds {
        bos: kv_u32!(kv_pairs, "tokenizer.ggml.bos_token_id"),
        eos: kv_u32!(kv_pairs, "tokenizer.ggml.eos_token_id"),
        unk: kv_u32!(kv_pairs, "tokenizer.ggml.unknown_token_id"),
        pad: kv_u32!(kv_pairs, "tokenizer.ggml.padding_token_id"),
        eot: kv_u32!(kv_pairs, "tokenizer.ggml.eot_token_id"),
        eog: kv_u32!(kv_pairs, "tokenizer.ggml.eog_token_id"),
        sep: kv_u32!(kv_pairs, "tokenizer.ggml.sep_token_id"),
        nl: kv_u32!(kv_pairs, "tokenizer.ggml.nl_token_id"),
        fim_pre: kv_u32!(kv_pairs, "tokenizer.ggml.fim_pre_token_id"),
        fim_suf: kv_u32!(kv_pairs, "tokenizer.ggml.fim_suf_token_id"),
        fim_mid: kv_u32!(kv_pairs, "tokenizer.ggml.fim_mid_token_id"),
        mask: kv_u32!(kv_pairs, "tokenizer.ggml.mask_token_id"),
    };

    // Tokenization flags — fall back to llama.cpp defaults when absent
    let flags = TokenizationFlags {
        add_bos_token: kv_bool!(kv_pairs, "tokenizer.ggml.add_bos_token").unwrap_or(true),
        add_eos_token: kv_bool!(kv_pairs, "tokenizer.ggml.add_eos_token").unwrap_or(false),
        add_space_prefix: kv_bool!(kv_pairs, "tokenizer.ggml.add_space_prefix").unwrap_or(true),
        clean_spaces: kv_bool!(kv_pairs, "tokenizer.ggml.clean_spaces").unwrap_or(false),
        remove_extra_whitespaces: kv_bool!(kv_pairs, "tokenizer.ggml.remove_extra_whitespaces")
            .unwrap_or(false),
        escape_whitespaces: kv_bool!(kv_pairs, "tokenizer.ggml.escape_whitespaces")
            .unwrap_or(false),
        treat_whitespace_as_suffix: kv_bool!(kv_pairs, "tokenizer.ggml.treat_whitespace_as_suffix")
            .unwrap_or(false),
    };

    // BPE merges
    let merges = match kv_pairs.get("tokenizer.ggml.merges") {
        Some(Value::StringArray(arr)) => {
            let mut result = Vec::new();
            for merge_str in arr {
                let parts: Vec<&str> = merge_str.split(' ').collect();
                if parts.len() == 2 {
                    result.push((parts[0].to_string(), parts[1].to_string()));
                }
            }
            Some(result)
        }
        _ => None,
    };

    Ok(GGUFMetadata {
        tokens,
        scores,
        token_types,
        model_type,
        pre_type,
        chat_template,
        special,
        flags,
        merges,
    })
}

/// A typed value read from a GGUF metadata key-value pair.
///
/// Only the variants needed to reconstruct a tokenizer are surfaced here.
/// Unrecognised type IDs return [`Error::InvalidMetadata`].
#[derive(Debug)]
enum Value {
    U32(u32),
    #[allow(dead_code)] // read via pattern-match in read_value; Debug confuses the lint
    I32(i32),
    #[allow(dead_code)] // read via pattern-match in read_value; Debug confuses the lint
    F32(f32),
    Bool(bool),
    #[allow(dead_code)] // Retained to consume GGUF BOOL arrays not used by tokenization.
    BoolArray(Vec<bool>),
    String(String),
    StringArray(Vec<String>),
    I32Array(Vec<i32>),
    F32Array(Vec<f32>),
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32, Error> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64, Error> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i32<R: Read>(reader: &mut R) -> Result<i32, Error> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_f32<R: Read>(reader: &mut R) -> Result<f32, Error> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_string<R: Read>(reader: &mut R, total_bytes: &mut usize) -> Result<String, Error> {
    const MAX_STRING_SIZE: usize = 1024 * 1024; // 1MB max per string
    const MAX_TOTAL_STRING_DATA: usize = 100 * 1024 * 1024; // 100MB total
    let len_u64 = read_u64(reader)?;

    // Prevent truncation on 32-bit systems (Issue R4#12)
    if len_u64 > usize::MAX as u64 {
        return Err(Error::InvalidMetadata(format!(
            "String length {len_u64} exceeds platform limit"
        )));
    }
    let len = len_u64 as usize;

    if len > MAX_STRING_SIZE {
        return Err(Error::InvalidMetadata(format!(
            "String too large: {len} bytes (max: {MAX_STRING_SIZE})"
        )));
    }

    // Check total allocation with overflow protection (Issue R3#2)
    *total_bytes = total_bytes
        .checked_add(len)
        .ok_or_else(|| Error::InvalidMetadata("Total string data overflow".to_string()))?;
    if *total_bytes > MAX_TOTAL_STRING_DATA {
        return Err(Error::InvalidMetadata(format!(
            "Total string data too large: {} bytes (max: {})",
            *total_bytes, MAX_TOTAL_STRING_DATA
        )));
    }

    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| Error::InvalidMetadata(format!("Invalid UTF-8: {e}")))
}

fn read_value<R: Read>(reader: &mut R, total_bytes: &mut usize) -> Result<Value, Error> {
    let type_id = read_u32(reader)?;

    match type_id {
        4 => Ok(Value::U32(read_u32(reader)?)),
        5 => Ok(Value::I32(read_i32(reader)?)),
        6 => Ok(Value::F32(read_f32(reader)?)),
        7 => {
            let mut byte = [0u8; 1];
            reader.read_exact(&mut byte)?;
            Ok(Value::Bool(byte[0] != 0))
        }
        8 => Ok(Value::String(read_string(reader, total_bytes)?)),
        9 => {
            // Array
            let array_type = read_u32(reader)?;
            let array_len = read_u64(reader)? as usize;

            match array_type {
                7 => {
                    // BOOL array: one byte per boolean in the GGUF wire format.
                    let mut arr = Vec::with_capacity(array_len);
                    for _ in 0..array_len {
                        let mut byte = [0u8; 1];
                        reader.read_exact(&mut byte)?;
                        arr.push(byte[0] != 0);
                    }
                    Ok(Value::BoolArray(arr))
                }
                5 => {
                    // I32 array
                    let mut arr = Vec::with_capacity(array_len);
                    for _ in 0..array_len {
                        arr.push(read_i32(reader)?);
                    }
                    Ok(Value::I32Array(arr))
                }
                6 => {
                    // F32 array
                    let mut arr = Vec::with_capacity(array_len);
                    for _ in 0..array_len {
                        arr.push(read_f32(reader)?);
                    }
                    Ok(Value::F32Array(arr))
                }
                8 => {
                    // String array
                    let mut arr = Vec::with_capacity(array_len);
                    for _ in 0..array_len {
                        arr.push(read_string(reader, total_bytes)?);
                    }
                    Ok(Value::StringArray(arr))
                }
                _ => Err(Error::InvalidMetadata(format!(
                    "Unsupported array type: {array_type}"
                ))),
            }
        }
        _ => Err(Error::InvalidMetadata(format!(
            "Unsupported value type: {type_id}"
        ))),
    }
}
