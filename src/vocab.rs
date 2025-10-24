//! Vocabulary loading and management from GGUF model files.
//!
//! This module handles:
//! - Loading token strings, scores, and types from GGUF metadata
//! - Managing BPE merge rules
//! - Providing token lookup (text → ID, ID → text)
//! - Tracking special tokens (BOS, EOS, UNK, etc.)
//!
//! # Validation
//! Performs safety checks at load time:
//! - Max vocab size: 1M tokens
//! - Max token length: 1KB per token
//! - Validates all arrays have consistent length
//!
//! # Special Token Handling
//! Supports llama.cpp's special token conventions:
//! - `<s>`, `</s>`: Beginning/end of sequence
//! - `<unk>`: Unknown token fallback
//! - Model-specific tokens via metadata

use crate::{Error, TokenId};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    Undefined = 0,
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    Byte = 6,
}

impl From<i32> for TokenType {
    fn from(value: i32) -> Self {
        match value {
            1 => TokenType::Normal,
            2 => TokenType::Unknown,
            3 => TokenType::Control,
            4 => TokenType::UserDefined,
            5 => TokenType::Unused,
            6 => TokenType::Byte,
            _ => TokenType::Undefined,
        }
    }
}

pub struct Vocabulary {
    tokens: Vec<String>,
    scores: Vec<f32>,
    token_types: Vec<TokenType>,
    token_to_id: HashMap<String, TokenId>,

    // Model metadata
    model_type: String,
    #[allow(dead_code)]
    pre_type: String,

    // Special tokens
    bos_token_id: TokenId,
    eos_token_id: TokenId,
    unk_token_id: TokenId,
    pad_token_id: Option<TokenId>,

    // Tokenization flags
    add_bos_token: bool,
    add_eos_token: bool,
    #[allow(dead_code)]
    add_space_prefix: bool,

    // For BPE models
    merges: Vec<(String, String)>,
}

impl Vocabulary {
    pub fn from_gguf_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let metadata = crate::gguf::load_metadata(path)?;

        const MAX_VOCAB_SIZE: usize = 1_000_000; // 1M tokens max
        const MAX_TOKEN_LENGTH: usize = 1024; // 1KB per token max

        let num_tokens = metadata.tokens.len();

        // Validate vocab size
        if num_tokens == 0 {
            return Err(Error::VocabularyError("Vocabulary is empty".to_string()));
        }
        if num_tokens > MAX_VOCAB_SIZE {
            return Err(Error::VocabularyError(format!(
                "Vocabulary too large: {num_tokens} tokens (max: {MAX_VOCAB_SIZE})"
            )));
        }

        // Validate token lengths
        for (i, token) in metadata.tokens.iter().enumerate() {
            if token.len() > MAX_TOKEN_LENGTH {
                return Err(Error::VocabularyError(format!(
                    "Token {} too large: {} bytes (max: {})",
                    i,
                    token.len(),
                    MAX_TOKEN_LENGTH
                )));
            }
        }

        // Build token_to_id with capacity hint (Issue #8)
        let mut token_to_id = HashMap::with_capacity(num_tokens);
        for (i, s) in metadata.tokens.iter().enumerate() {
            token_to_id.insert(s.clone(), i as TokenId);
        }

        // Validate no duplicate tokens
        if token_to_id.len() != num_tokens {
            return Err(Error::VocabularyError(format!(
                "Duplicate tokens found: {} unique out of {} total",
                token_to_id.len(),
                num_tokens
            )));
        }

        // Validate merge rules reference valid tokens (Issue #12)
        if let Some(ref merges) = metadata.merges {
            const MAX_MERGE_COUNT: usize = 1_000_000; // 1M merges max (Issue R3#14)
            if merges.len() > MAX_MERGE_COUNT {
                return Err(Error::VocabularyError(format!(
                    "Too many merge rules: {} (max: {})",
                    merges.len(),
                    MAX_MERGE_COUNT
                )));
            }

            for (rank, (left, right)) in merges.iter().enumerate() {
                if !token_to_id.contains_key(left) {
                    return Err(Error::VocabularyError(format!(
                        "Merge rule {rank} references unknown left token: '{left}'"
                    )));
                }
                if !token_to_id.contains_key(right) {
                    return Err(Error::VocabularyError(format!(
                        "Merge rule {rank} references unknown right token: '{right}'"
                    )));
                }
            }
        }

        let scores = metadata.scores.unwrap_or_else(|| vec![0.0; num_tokens]);

        // Validate scores length matches tokens (Issue R3#11)
        if scores.len() != num_tokens {
            return Err(Error::VocabularyError(format!(
                "Score array length mismatch: {} scores for {} tokens",
                scores.len(),
                num_tokens
            )));
        }

        Ok(Self {
            tokens: metadata.tokens,
            scores,
            token_types: {
                let types = metadata
                    .token_types
                    .unwrap_or_else(|| vec![TokenType::Normal; num_tokens]);
                // Validate token_types length (Issue R3#11)
                if types.len() != num_tokens {
                    return Err(Error::VocabularyError(format!(
                        "Token types length mismatch: {} types for {} tokens",
                        types.len(),
                        num_tokens
                    )));
                }
                types
            },
            token_to_id,

            model_type: metadata.model_type,
            pre_type: metadata.pre_type.unwrap_or_else(|| "default".to_string()),

            bos_token_id: metadata.bos_token_id.unwrap_or(1),
            eos_token_id: metadata.eos_token_id.unwrap_or(2),
            unk_token_id: metadata.unk_token_id.unwrap_or(0),
            pad_token_id: metadata.pad_token_id,

            add_bos_token: metadata.add_bos_token.unwrap_or(true),
            add_eos_token: metadata.add_eos_token.unwrap_or(false),
            add_space_prefix: metadata.add_space_prefix.unwrap_or(true),

            merges: metadata.merges.unwrap_or_default(),
        })
    }

    #[must_use] 
    pub fn model_type(&self) -> &str {
        &self.model_type
    }

    #[must_use] 
    pub fn get_token_id(&self, text: &str) -> Option<TokenId> {
        self.token_to_id.get(text).copied()
    }

    #[must_use] 
    pub fn get_token_text(&self, id: TokenId) -> Option<&str> {
        self.tokens.get(id as usize).map(std::string::String::as_str)
    }

    #[must_use] 
    pub fn get_token_score(&self, id: TokenId) -> f32 {
        // scores length is validated to match tokens, so this is safe
        self.scores
            .get(id as usize)
            .copied()
            .expect("Token ID should be valid - scores validated at load time")
    }

    #[must_use] 
    pub fn get_token_type(&self, id: TokenId) -> TokenType {
        // token_types length is validated to match tokens, so this is safe
        self.token_types
            .get(id as usize)
            .copied()
            .expect("Token ID should be valid - token_types validated at load time")
    }

    #[must_use] 
    pub fn byte_to_token(&self, byte: u8) -> TokenId {
        // Try hex format <0xXX> first (SPM style)
        let hex_str = format!("<0x{byte:02X}>");
        if let Some(id) = self.token_to_id.get(&hex_str) {
            return *id;
        }

        // Fall back to raw byte as string
        let byte_str = String::from_utf8_lossy(&[byte]).to_string();
        self.token_to_id
            .get(&byte_str)
            .copied()
            .unwrap_or(self.unk_token_id)
    }

    #[must_use] 
    pub fn is_special_token(&self, id: TokenId) -> bool {
        matches!(
            self.get_token_type(id),
            TokenType::Control | TokenType::Unknown
        ) || id == self.bos_token_id
            || id == self.eos_token_id
            || id == self.unk_token_id
            || self.pad_token_id == Some(id)
    }

    #[must_use] 
    pub fn add_bos_token(&self) -> bool {
        self.add_bos_token
    }

    #[must_use] 
    pub fn add_eos_token(&self) -> bool {
        self.add_eos_token
    }

    #[must_use] 
    pub fn bos_token_id(&self) -> TokenId {
        self.bos_token_id
    }

    #[must_use] 
    pub fn eos_token_id(&self) -> TokenId {
        self.eos_token_id
    }

    #[must_use] 
    pub fn unk_token_id(&self) -> TokenId {
        self.unk_token_id
    }

    #[must_use] 
    pub fn get_merges(&self) -> &[(String, String)] {
        &self.merges
    }

    #[must_use] 
    pub fn pre_type(&self) -> Option<&str> {
        if self.pre_type.is_empty() || self.pre_type == "default" {
            None
        } else {
            Some(&self.pre_type)
        }
    }

    #[must_use] 
    pub fn n_tokens(&self) -> usize {
        self.tokens.len()
    }
}
