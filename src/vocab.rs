//! GGUF vocabulary loading and management

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
                "Vocabulary too large: {} tokens (max: {})",
                num_tokens, MAX_VOCAB_SIZE
            )));
        }

        // Validate token lengths
        for (i, token) in metadata.tokens.iter().enumerate() {
            if token.len() > MAX_TOKEN_LENGTH {
                return Err(Error::VocabularyError(format!(
                    "Token {} too large: {} bytes (max: {})",
                    i, token.len(), MAX_TOKEN_LENGTH
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
                token_to_id.len(), num_tokens
            )));
        }

        // Validate merge rules reference valid tokens (Issue #12)
        if let Some(ref merges) = metadata.merges {
            for (rank, (left, right)) in merges.iter().enumerate() {
                if !token_to_id.contains_key(left) {
                    return Err(Error::VocabularyError(format!(
                        "Merge rule {} references unknown left token: '{}'",
                        rank, left
                    )));
                }
                if !token_to_id.contains_key(right) {
                    return Err(Error::VocabularyError(format!(
                        "Merge rule {} references unknown right token: '{}'",
                        rank, right
                    )));
                }
            }
        }

        Ok(Self {
            tokens: metadata.tokens,
            scores: metadata.scores.unwrap_or_else(|| vec![0.0; num_tokens]),
            token_types: metadata
                .token_types
                .unwrap_or_else(|| vec![TokenType::Normal; num_tokens]),
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

    pub fn model_type(&self) -> &str {
        &self.model_type
    }

    pub fn get_token_id(&self, text: &str) -> Option<TokenId> {
        self.token_to_id.get(text).copied()
    }

    pub fn get_token_text(&self, id: TokenId) -> Option<&str> {
        self.tokens.get(id as usize).map(|s| s.as_str())
    }

    pub fn get_token_score(&self, id: TokenId) -> f32 {
        self.scores.get(id as usize).copied().unwrap_or(0.0)
    }

    pub fn get_token_type(&self, id: TokenId) -> TokenType {
        self.token_types
            .get(id as usize)
            .copied()
            .unwrap_or(TokenType::Undefined)
    }

    pub fn byte_to_token(&self, byte: u8) -> TokenId {
        // Try hex format <0xXX> first (SPM style)
        let hex_str = format!("<0x{:02X}>", byte);
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

    pub fn is_special_token(&self, id: TokenId) -> bool {
        matches!(
            self.get_token_type(id),
            TokenType::Control | TokenType::Unknown
        ) || id == self.bos_token_id
            || id == self.eos_token_id
            || id == self.unk_token_id
            || self.pad_token_id == Some(id)
    }

    pub fn add_bos_token(&self) -> bool {
        self.add_bos_token
    }

    pub fn add_eos_token(&self) -> bool {
        self.add_eos_token
    }

    pub fn bos_token_id(&self) -> TokenId {
        self.bos_token_id
    }

    pub fn eos_token_id(&self) -> TokenId {
        self.eos_token_id
    }

    pub fn unk_token_id(&self) -> TokenId {
        self.unk_token_id
    }

    pub fn get_merges(&self) -> &[(String, String)] {
        &self.merges
    }

    pub fn pre_type(&self) -> Option<&str> {
        if self.pre_type.is_empty() || self.pre_type == "default" {
            None
        } else {
            Some(&self.pre_type)
        }
    }

    pub fn n_tokens(&self) -> usize {
        self.tokens.len()
    }
}
