//! # shimmytok
//!
//! Pure Rust tokenizer for GGUF models with llama.cpp compatibility.
//!
//! ## Features
//!
//! - 🦀 Pure Rust - no C++ dependencies
//! - 📦 Load tokenizers directly from GGUF files
//! - ✅ 100% compatible with llama.cpp
//! - 🧪 Fully tested against llama.cpp output
//! - 🎯 Simple API - 3 methods
//!
//! ## Example
//!
//! ```no_run
//! use shimmytok::Tokenizer;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load tokenizer from GGUF file
//! let tokenizer = Tokenizer::from_gguf_file("model.gguf")?;
//!
//! // Encode text to token IDs
//! let tokens = tokenizer.encode("Hello world", true)?;
//!
//! // Decode token IDs back to text
//! let text = tokenizer.decode(&tokens, true)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Supported Models
//!
//! - ✅ LLaMA / Llama-2 / Llama-3 (SentencePiece)
//! - ✅ Phi-3 (SentencePiece)
//! - ✅ GPT-2 / GPT-3 style BPE

use std::path::Path;

pub mod bpe;
pub mod byte_encoder;
pub mod gguf;
pub mod sentencepiece;
pub mod vocab;

pub use vocab::{TokenType, Vocabulary};

/// Token ID type used throughout the library
/// Type alias for token IDs
///
/// Token IDs are represented as u32 to match GGUF format and llama.cpp implementation.
/// This is safe because vocabulary size is limited to MAX_VOCAB_SIZE (1M tokens),
/// which is well below u32::MAX (4.2B). (Issue R2#10)
pub type TokenId = u32;

/// Main tokenizer interface for encoding and decoding text
///
/// The tokenizer loads vocabulary and configuration from GGUF files and provides
/// methods to convert between text and token IDs.
///
/// # Example
///
/// ```no_run
/// use shimmytok::Tokenizer;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let tokenizer = Tokenizer::from_gguf_file("model.gguf")?;
/// let tokens = tokenizer.encode("Hello world", true)?;
/// let text = tokenizer.decode(&tokens, true)?;
/// # Ok(())
/// # }
/// ```
pub struct Tokenizer {
    vocab: Vocabulary,
    tokenizer_impl: Box<dyn TokenizerImpl>,
}

trait TokenizerImpl {
    fn encode(&self, text: &str, vocab: &Vocabulary) -> Result<Vec<TokenId>, Error>;
    fn decode(&self, tokens: &[TokenId], vocab: &Vocabulary) -> Result<String, Error>;
}

impl Tokenizer {
    /// Load a tokenizer from a GGUF model file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the GGUF file containing model and tokenizer data
    ///
    /// # Returns
    ///
    /// Returns `Ok(Tokenizer)` on success, or `Err(Error)` if the file cannot be read,
    /// is not a valid GGUF file, or contains an unsupported tokenizer type.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use shimmytok::Tokenizer;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let tokenizer = Tokenizer::from_gguf_file("model.gguf")?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "from_gguf_file returns a Result that must be handled"]
    pub fn from_gguf_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let vocab = Vocabulary::from_gguf_file(path)?;

        let tokenizer_impl: Box<dyn TokenizerImpl> = match vocab.model_type() {
            "llama" => Box::new(sentencepiece::SentencePieceTokenizer::new()),
            "gpt2" => Box::new(bpe::BPETokenizer::new()),
            model => return Err(Error::UnsupportedModel(model.to_string())),
        };

        Ok(Self {
            vocab,
            tokenizer_impl,
        })
    }

    /// Encode text into a sequence of token IDs
    ///
    /// # Arguments
    ///
    /// * `text` - The input text to tokenize
    /// * `add_special_tokens` - If true, adds BOS/EOS tokens according to model configuration
    ///
    /// # Returns
    ///
    /// Returns a vector of token IDs representing the input text.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use shimmytok::Tokenizer;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let tokenizer = Tokenizer::from_gguf_file("model.gguf")?;
    /// let tokens = tokenizer.encode("Hello world", true)?;
    /// println!("Tokens: {:?}", tokens);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "encode returns a Result that must be handled"]
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<TokenId>, Error> {
        let mut tokens = Vec::new();

        if add_special_tokens && self.vocab.add_bos_token() {
            tokens.push(self.vocab.bos_token_id());
        }

        let encoded = self.tokenizer_impl.encode(text, &self.vocab)?;
        tokens.extend(encoded);

        if add_special_tokens && self.vocab.add_eos_token() {
            tokens.push(self.vocab.eos_token_id());
        }

        Ok(tokens)
    }

    /// Decode a sequence of token IDs back into text
    ///
    /// # Arguments
    ///
    /// * `tokens` - Slice of token IDs to decode
    /// * `skip_special_tokens` - If true, filters out special tokens (BOS, EOS, etc.)
    ///
    /// # Returns
    ///
    /// Returns the decoded text as a String.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use shimmytok::Tokenizer;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let tokenizer = Tokenizer::from_gguf_file("model.gguf")?;
    /// let tokens = vec![15043, 3186]; // "Hello world"
    /// let text = tokenizer.decode(&tokens, true)?;
    /// println!("Text: {}", text);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "decode returns a Result that must be handled"]
    pub fn decode(&self, tokens: &[TokenId], skip_special_tokens: bool) -> Result<String, Error> {
        let filtered_tokens = if skip_special_tokens {
            tokens
                .iter()
                .copied()
                .filter(|&id| !self.vocab.is_special_token(id))
                .collect::<Vec<_>>()
        } else {
            tokens.to_vec()
        };

        self.tokenizer_impl.decode(&filtered_tokens, &self.vocab)
    }

    /// Get the vocabulary size
    ///
    /// # Returns
    ///
    /// The total number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.vocab.n_tokens()
    }

    /// Get the Beginning-of-Sequence (BOS) token ID
    ///
    /// # Returns
    ///
    /// The token ID used to mark the beginning of a sequence.
    pub fn bos_token(&self) -> TokenId {
        self.vocab.bos_token_id()
    }

    /// Get the End-of-Sequence (EOS) token ID
    ///
    /// # Returns
    ///
    /// The token ID used to mark the end of a sequence.
    pub fn eos_token(&self) -> TokenId {
        self.vocab.eos_token_id()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Failed to read GGUF file: {0}")]
    GGUFRead(String),

    #[error("Invalid GGUF metadata: {0}")]
    InvalidMetadata(String),

    #[error("Unsupported model type: {0}")]
    UnsupportedModel(String),

    #[error("Tokenization failed: {0}")]
    TokenizationFailed(String),

    #[error("Invalid token: {0}")]
    InvalidToken(String),

    #[error("Vocabulary error: {0}")]
    VocabularyError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
