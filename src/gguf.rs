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

/// A typed value preserved from a GGUF metadata key-value pair.
#[derive(Debug, Clone, PartialEq)]
pub enum GGUFValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Self>),
}

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
    /// All metadata values, including keys not used by the tokenizer.
    metadata: HashMap<String, GGUFValue>,
}

impl GGUFMetadata {
    /// Returns all GGUF metadata values as a read-only map.
    #[must_use]
    pub fn metadata(&self) -> &HashMap<String, GGUFValue> {
        &self.metadata
    }
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
pub fn load_metadata_from_reader<R: Read>(reader: R) -> Result<GGUFMetadata, Error> {
    /// Extract an optional `u32` from a kv-pair map.
    macro_rules! kv_u32 {
        ($map:expr, $key:expr) => {
            match $map.get($key) {
                Some(GGUFValue::U32(v)) => Some(*v),
                _ => None,
            }
        };
    }

    /// Extract an optional `bool` from a kv-pair map.
    macro_rules! kv_bool {
        ($map:expr, $key:expr) => {
            match $map.get($key) {
                Some(GGUFValue::Bool(v)) => Some(*v),
                _ => None,
            }
        };
    }

    let parsed = crate::manifest::read_metadata(reader, crate::manifest::ParserLimits::default())?;
    let kv_pairs = parsed.values;
    let metadata = kv_pairs.clone();

    // Extract tokenizer metadata
    let tokens = match kv_pairs.get("tokenizer.ggml.tokens") {
        Some(GGUFValue::Array(arr))
            if parsed.array_element_types.get("tokenizer.ggml.tokens") == Some(&8) =>
        {
            arr.iter()
                .map(|v| match v {
                    GGUFValue::String(s) => s.clone(),
                    _ => unreachable!("shared parser array type"),
                })
                .collect()
        }
        _ => {
            return Err(Error::InvalidMetadata(
                "Missing tokenizer.ggml.tokens".into(),
            ))
        }
    };

    let scores = match kv_pairs.get("tokenizer.ggml.scores") {
        Some(GGUFValue::Array(arr))
            if parsed.array_element_types.get("tokenizer.ggml.scores") == Some(&6) =>
        {
            Some(
                arr.iter()
                    .map(|v| match v {
                        GGUFValue::F32(x) => *x,
                        _ => unreachable!("shared parser array type"),
                    })
                    .collect(),
            )
        }
        _ => None,
    };

    let token_types = match kv_pairs.get("tokenizer.ggml.token_type") {
        Some(GGUFValue::Array(arr))
            if parsed.array_element_types.get("tokenizer.ggml.token_type") == Some(&5) =>
        {
            Some(
                arr.iter()
                    .map(|v| match v {
                        GGUFValue::I32(x) => TokenType::from(*x),
                        _ => unreachable!("shared parser array type"),
                    })
                    .collect(),
            )
        }
        _ => None,
    };

    let model_type = match kv_pairs.get("tokenizer.ggml.model") {
        Some(GGUFValue::String(s)) => s.clone(),
        _ => "llama".to_string(), // Default
    };

    let pre_type = match kv_pairs.get("tokenizer.ggml.pre") {
        Some(GGUFValue::String(s)) => Some(s.clone()),
        _ => None,
    };

    // Chat template — raw Jinja2 string, pass to shimmyjinja to render prompts
    let chat_template = match kv_pairs.get("tokenizer.chat_template") {
        Some(GGUFValue::String(s)) => Some(s.clone()),
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

    // Tokenization flags — fall back to llama.cpp defaults when absent.
    // llama.cpp's add_bos_token default is true only for SentencePiece
    // tokenizers (llama/mistral/gemma); BPE/WPM families default to false.
    // An explicit key always wins. Models like Qwen2 and MiniCPM-V omit the
    // key entirely, so defaulting true there poisons prompts with a BOS.
    let default_add_bos = matches!(
        model_type.as_str(),
        "llama" | "mistral" | "gemma" | "gemma4" | "gemma6"
    );
    let flags = TokenizationFlags {
        add_bos_token: kv_bool!(kv_pairs, "tokenizer.ggml.add_bos_token")
            .unwrap_or(default_add_bos),
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
        Some(GGUFValue::Array(arr))
            if parsed.array_element_types.get("tokenizer.ggml.merges") == Some(&8) =>
        {
            let mut result = Vec::new();
            for value in arr {
                let GGUFValue::String(merge_str) = value else {
                    unreachable!("shared parser array type")
                };
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
        metadata,
    })
}
