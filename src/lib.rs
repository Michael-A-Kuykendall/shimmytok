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
//! - ✅ `LLaMA` / Llama-2 / Llama-3 (`SentencePiece`)
//! - ✅ Mistral (`SentencePiece`)
//! - ✅ Phi-3 (`SentencePiece`)
//! - ✅ Qwen / Qwen2 (BPE)
//! - ✅ Gemma (`SentencePiece`)
//! - ✅ GPT-2 / GPT-3 style BPE

use rayon::prelude::*;
use std::path::Path;

pub mod bpe;
pub mod byte_encoder;
pub mod gguf;
pub mod plamo2;
pub mod rwkv;
pub mod sentencepiece;
pub mod ugm;
pub mod vocab;
pub mod wpm;

pub use plamo2::Plamo2Tokenizer;
pub use rwkv::RwkvTokenizer;
pub use ugm::UgmTokenizer;
pub use vocab::{TokenType, Vocabulary};
pub use wpm::WpmTokenizer;

/// Token ID type used throughout the library
/// Maximum input text size in bytes (10MB) - Issue R4#2
pub const MAX_INPUT_SIZE: usize = 10 * 1024 * 1024;

/// Maximum output tokens (1M tokens max) - prevents memory exhaustion
pub const MAX_OUTPUT_TOKENS: usize = 1_000_000;

/// Options for encoding text (llama.cpp parity)
#[derive(Debug, Clone, Default)]
pub struct EncodeOptions {
    /// Add BOS/EOS tokens according to model configuration
    pub add_special_tokens: bool,
    /// Parse special token strings in input (e.g., `<|eot_id|>`) and emit as tokens
    pub parse_special: bool,
}

impl EncodeOptions {
    /// Create options with add_special_tokens only (legacy behavior)
    #[must_use]
    pub fn with_special_tokens(add_special_tokens: bool) -> Self {
        Self {
            add_special_tokens,
            parse_special: false,
        }
    }

    /// Create options that parse special tokens in input
    #[must_use]
    pub fn with_parse_special(add_special_tokens: bool, parse_special: bool) -> Self {
        Self {
            add_special_tokens,
            parse_special,
        }
    }
}

/// Options for decoding tokens (llama.cpp parity)
#[derive(Debug, Clone, Default)]
pub struct DecodeOptions {
    /// Skip special tokens (BOS, EOS, etc.) in output
    pub skip_special_tokens: bool,
    /// Strip leading whitespace from each token piece
    pub lstrip: bool,
    /// If false, emit empty string for special/control tokens instead of their text
    pub include_special_text: bool,
}

impl DecodeOptions {
    /// Create options with skip_special_tokens only (legacy behavior)
    #[must_use]
    pub fn with_skip_special(skip_special_tokens: bool) -> Self {
        Self {
            skip_special_tokens,
            lstrip: false,
            include_special_text: true,
        }
    }

    /// Create full decode options
    #[must_use]
    pub fn new(skip_special_tokens: bool, lstrip: bool, include_special_text: bool) -> Self {
        Self {
            skip_special_tokens,
            lstrip,
            include_special_text,
        }
    }
}

/// Type alias for token IDs
///
/// Token IDs are represented as u32 to match GGUF format and llama.cpp implementation.
/// This is safe because vocabulary size is limited to `MAX_VOCAB_SIZE` (1M tokens),
/// which is well below `u32::MAX` (4.2B). (Issue R2#10)
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

trait TokenizerImpl: Send + Sync {
    fn encode(&self, text: &str, vocab: &Vocabulary) -> Result<Vec<TokenId>, Error>;
    fn decode(&self, tokens: &[TokenId], vocab: &Vocabulary) -> Result<String, Error>;
}

// Wrapper for WPM tokenizer to implement TokenizerImpl
struct WpmWrapper {
    inner: wpm::WpmTokenizer,
}

impl TokenizerImpl for WpmWrapper {
    fn encode(&self, text: &str, vocab: &Vocabulary) -> Result<Vec<TokenId>, Error> {
        self.inner.encode(text, vocab)
    }
    fn decode(&self, tokens: &[TokenId], vocab: &Vocabulary) -> Result<String, Error> {
        self.inner.decode(tokens, vocab)
    }
}

// Wrapper for RWKV tokenizer to implement TokenizerImpl
struct RwkvWrapper {
    inner: rwkv::RwkvTokenizer,
}

impl TokenizerImpl for RwkvWrapper {
    fn encode(&self, text: &str, vocab: &Vocabulary) -> Result<Vec<TokenId>, Error> {
        self.inner.encode(text, vocab)
    }
    fn decode(&self, tokens: &[TokenId], vocab: &Vocabulary) -> Result<String, Error> {
        self.inner.decode(tokens, vocab)
    }
}

// Wrapper for UGM tokenizer to implement TokenizerImpl
struct UgmWrapper {
    inner: ugm::UgmTokenizer,
}

impl TokenizerImpl for UgmWrapper {
    fn encode(&self, text: &str, vocab: &Vocabulary) -> Result<Vec<TokenId>, Error> {
        self.inner.encode(text, vocab)
    }
    fn decode(&self, tokens: &[TokenId], vocab: &Vocabulary) -> Result<String, Error> {
        self.inner.decode(tokens, vocab)
    }
}

// Wrapper for PLaMo-2 tokenizer to implement TokenizerImpl
struct Plamo2Wrapper {
    inner: plamo2::Plamo2Tokenizer,
}

impl TokenizerImpl for Plamo2Wrapper {
    fn encode(&self, text: &str, vocab: &Vocabulary) -> Result<Vec<TokenId>, Error> {
        self.inner.encode(text, vocab)
    }
    fn decode(&self, tokens: &[TokenId], vocab: &Vocabulary) -> Result<String, Error> {
        self.inner.decode(tokens, vocab)
    }
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
            // SentencePiece models
            "llama" => Box::new(sentencepiece::SentencePieceTokenizer::new()),
            "mistral" => Box::new(sentencepiece::SentencePieceTokenizer::new()),
            "gemma" => Box::new(sentencepiece::SentencePieceTokenizer::new()),
            // BPE models
            "gpt2" => Box::new(bpe::BPETokenizer::new()),
            "qwen" | "qwen2" => Box::new(bpe::BPETokenizer::new()),
            // WPM (WordPiece) models - BERT-style
            "bert" | "wpm" => Box::new(WpmWrapper {
                inner: wpm::WpmTokenizer::new(&vocab),
            }),
            // RWKV models - trie-based greedy
            "rwkv" => Box::new(RwkvWrapper {
                inner: rwkv::RwkvTokenizer::new(&vocab),
            }),
            // UGM (Unigram) models - T5-style Viterbi
            "t5" | "ugm" => Box::new(UgmWrapper {
                inner: ugm::UgmTokenizer::new(&vocab),
            }),
            // PLaMo-2 models - table-driven DP
            "plamo2" => Box::new(Plamo2Wrapper {
                inner: plamo2::Plamo2Tokenizer::new(&vocab)?,
            }),
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
        self.encode_with_options(
            text,
            &EncodeOptions::with_special_tokens(add_special_tokens),
        )
    }

    /// Encode text into a sequence of token IDs with full options
    ///
    /// # Arguments
    ///
    /// * `text` - The input text to tokenize
    /// * `options` - Encoding options (add_special_tokens, parse_special)
    ///
    /// # Returns
    ///
    /// Returns a vector of token IDs representing the input text.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use shimmytok::{Tokenizer, EncodeOptions};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let tokenizer = Tokenizer::from_gguf_file("model.gguf")?;
    ///
    /// // Parse special tokens like <|eot_id|> in input
    /// let opts = EncodeOptions::with_parse_special(true, true);
    /// let tokens = tokenizer.encode_with_options("Hello<|eot_id|>World", &opts)?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "encode_with_options returns a Result that must be handled"]
    pub fn encode_with_options(
        &self,
        text: &str,
        options: &EncodeOptions,
    ) -> Result<Vec<TokenId>, Error> {
        let mut tokens = Vec::new();

        if options.add_special_tokens && self.vocab.add_bos_token() {
            tokens.push(self.vocab.bos_token_id());
        }

        if options.parse_special {
            // Build special token map and find occurrences in text
            let special_map = self.vocab.special_token_map();
            let fragments = split_on_special_tokens(text, &special_map);

            for fragment in fragments {
                match fragment {
                    TextFragment::Special(token_id) => {
                        tokens.push(token_id);
                    }
                    TextFragment::Text(t) => {
                        if !t.is_empty() {
                            let encoded = self.tokenizer_impl.encode(&t, &self.vocab)?;
                            tokens.extend(encoded);
                        }
                    }
                }
            }
        } else {
            let encoded = self.tokenizer_impl.encode(text, &self.vocab)?;
            tokens.extend(encoded);
        }

        if options.add_special_tokens && self.vocab.add_eos_token() {
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
        self.decode_with_options(
            tokens,
            &DecodeOptions::with_skip_special(skip_special_tokens),
        )
    }

    /// Decode a sequence of token IDs back into text with full options
    ///
    /// # Arguments
    ///
    /// * `tokens` - Slice of token IDs to decode
    /// * `options` - Decoding options (skip_special_tokens, lstrip, include_special_text)
    ///
    /// # Returns
    ///
    /// Returns the decoded text as a String.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use shimmytok::{Tokenizer, DecodeOptions};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let tokenizer = Tokenizer::from_gguf_file("model.gguf")?;
    /// let tokens = vec![15043, 3186];
    ///
    /// // Decode with lstrip to remove leading whitespace from tokens
    /// let opts = DecodeOptions::new(true, true, false);
    /// let text = tokenizer.decode_with_options(&tokens, &opts)?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "decode_with_options returns a Result that must be handled"]
    pub fn decode_with_options(
        &self,
        tokens: &[TokenId],
        options: &DecodeOptions,
    ) -> Result<String, Error> {
        // Filter tokens based on options
        let filtered_tokens: Vec<TokenId> = if options.skip_special_tokens {
            tokens
                .iter()
                .copied()
                .filter(|&id| !self.vocab.is_special_token(id))
                .collect()
        } else {
            tokens.to_vec()
        };

        // If we need special handling (lstrip or include_special_text=false),
        // we need to decode token by token
        let mut result = if options.lstrip || !options.include_special_text {
            let mut result = String::new();
            for &token_id in &filtered_tokens {
                // Skip special text if requested
                if !options.include_special_text && self.vocab.is_special_token(token_id) {
                    continue;
                }

                // Get the token piece
                let piece = self.tokenizer_impl.decode(&[token_id], &self.vocab)?;

                // Apply lstrip if requested
                let piece = if options.lstrip {
                    piece.trim_start().to_string()
                } else {
                    piece
                };

                result.push_str(&piece);
            }
            result
        } else {
            // Standard decode path
            self.tokenizer_impl.decode(&filtered_tokens, &self.vocab)?
        };

        // Apply clean_spaces post-processing if enabled in vocab (llama.cpp parity)
        if self.vocab.clean_spaces() {
            result = apply_clean_spaces(&result);
        }

        Ok(result)
    }

    /// Get the vocabulary size
    ///
    /// # Returns
    ///
    /// The total number of tokens in the vocabulary.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab.n_tokens()
    }

    /// Get the Beginning-of-Sequence (BOS) token ID
    ///
    /// # Returns
    ///
    /// The token ID used to mark the beginning of a sequence.
    #[must_use]
    pub fn bos_token(&self) -> TokenId {
        self.vocab.bos_token_id()
    }

    /// Get the End-of-Sequence (EOS) token ID
    ///
    /// # Returns
    ///
    /// The token ID used to mark the end of a sequence.
    #[must_use]
    pub fn eos_token(&self) -> TokenId {
        self.vocab.eos_token_id()
    }

    /// Get the tokenizer model type
    ///
    /// Returns the model type identifier from the GGUF metadata.
    ///
    /// # Returns
    ///
    /// The model type string, such as "llama", "mistral", "gpt2", "qwen", "qwen2", or "gemma".
    ///
    /// # Example
    ///
    /// ```no_run
    /// use shimmytok::Tokenizer;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let tokenizer = Tokenizer::from_gguf_file("model.gguf")?;
    /// println!("Model type: {}", tokenizer.model_type());
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn model_type(&self) -> &str {
        self.vocab.model_type()
    }

    /// Get the pre-tokenization type (for BPE models)
    ///
    /// Returns the pre-tokenizer identifier from GGUF metadata.
    /// Used internally to select the correct regex patterns for pre-tokenization.
    ///
    /// # Returns
    ///
    /// The pre-type string like "llama3", "gpt-2", "deepseek-coder", etc., or None if not set.
    #[must_use]
    pub fn pre_type(&self) -> Option<&str> {
        self.vocab.pre_type()
    }

    /// Encode multiple texts in parallel
    ///
    /// This method uses parallel processing to encode multiple texts simultaneously,
    /// providing significant speedup for batch operations (typically 2-4x on multi-core systems).
    ///
    /// # Arguments
    ///
    /// * `texts` - Slice of text strings to tokenize
    /// * `add_special_tokens` - If true, adds BOS/EOS tokens according to model configuration
    ///
    /// # Returns
    ///
    /// Returns a vector of token ID vectors, one for each input text.
    /// The order of outputs matches the order of inputs.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use shimmytok::Tokenizer;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let tokenizer = Tokenizer::from_gguf_file("model.gguf")?;
    /// let texts = vec!["Hello world", "Goodbye world"];
    /// let batch = tokenizer.encode_batch(&texts, true)?;
    /// for (text, tokens) in texts.iter().zip(batch.iter()) {
    ///     println!("{}: {:?}", text, tokens);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "encode_batch returns a Result that must be handled"]
    pub fn encode_batch(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> Result<Vec<Vec<TokenId>>, Error> {
        texts
            .par_iter()
            .map(|text| self.encode(text, add_special_tokens))
            .collect()
    }

    /// Decode a single token to text
    ///
    /// This is useful for streaming generation where tokens are produced one at a time.
    /// Unlike `decode()`, this method handles single tokens more efficiently and is
    /// optimized for real-time streaming use cases.
    ///
    /// # Arguments
    ///
    /// * `token` - The token ID to decode
    /// * `skip_special_tokens` - If true, returns empty string for special tokens
    ///
    /// # Returns
    ///
    /// Returns the decoded text for this token.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use shimmytok::Tokenizer;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let tokenizer = Tokenizer::from_gguf_file("model.gguf")?;
    ///
    /// // Streaming generation simulation
    /// let tokens = vec![1, 15043, 3186]; // BOS, "Hello", "world"
    /// for token in tokens {
    ///     let text = tokenizer.decode_single(token, true)?;
    ///     print!("{}", text); // Prints: Helloworld
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "decode_single returns a Result that must be handled"]
    pub fn decode_single(
        &self,
        token: TokenId,
        skip_special_tokens: bool,
    ) -> Result<String, Error> {
        if skip_special_tokens && self.vocab.is_special_token(token) {
            return Ok(String::new());
        }
        self.tokenizer_impl.decode(&[token], &self.vocab)
    }

    /// Get the text representation of a token
    ///
    /// Returns the raw token piece (vocabulary entry) for a given token ID.
    /// This is useful for debugging and introspection.
    ///
    /// # Arguments
    ///
    /// * `token` - The token ID to look up
    ///
    /// # Returns
    ///
    /// Returns the token text, or an error if the token ID is invalid.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use shimmytok::Tokenizer;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let tokenizer = Tokenizer::from_gguf_file("model.gguf")?;
    /// let piece = tokenizer.token_to_piece(15043)?; // "Hello"
    /// println!("Token piece: {}", piece);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "token_to_piece returns a Result that must be handled"]
    pub fn token_to_piece(&self, token: TokenId) -> Result<String, Error> {
        self.vocab
            .get_token_text(token)
            .map(String::from)
            .ok_or_else(|| Error::InvalidToken(format!("Token ID {token} out of range")))
    }

    /// Get the type of a token
    ///
    /// Returns the token type classification from the vocabulary.
    ///
    /// # Arguments
    ///
    /// * `token` - The token ID to query
    ///
    /// # Returns
    ///
    /// Returns the token type, or `TokenType::Undefined` if the token ID is invalid.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use shimmytok::Tokenizer;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let tokenizer = Tokenizer::from_gguf_file("model.gguf")?;
    /// let token_type = tokenizer.token_type(1); // BOS token
    /// println!("Token type: {:?}", token_type);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn token_type(&self, token: TokenId) -> TokenType {
        if token >= self.vocab.n_tokens() as TokenId {
            return TokenType::Undefined;
        }
        self.vocab.get_token_type(token)
    }

    /// Check if a token is a special token
    ///
    /// Returns true if the token is a special token (BOS, EOS, UNK, PAD, or Control type).
    ///
    /// # Arguments
    ///
    /// * `token` - The token ID to check
    ///
    /// # Returns
    ///
    /// Returns true if the token is special, false otherwise.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use shimmytok::Tokenizer;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let tokenizer = Tokenizer::from_gguf_file("model.gguf")?;
    /// let is_special = tokenizer.is_special_token(1); // BOS token
    /// println!("Is special: {}", is_special); // true
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn is_special_token(&self, token: TokenId) -> bool {
        if token >= self.vocab.n_tokens() as TokenId {
            return false;
        }
        self.vocab.is_special_token(token)
    }
}

// ============================================================================
// Helper types and functions for parse_special mode
// ============================================================================

/// Fragment of text during parse_special tokenization
enum TextFragment {
    /// A special token that should be emitted directly
    Special(TokenId),
    /// Regular text that needs normal tokenization
    Text(String),
}

/// Split text on special token occurrences.
/// Returns fragments in order: either special tokens or regular text chunks.
fn split_on_special_tokens(
    text: &str,
    special_map: &std::collections::HashMap<String, TokenId>,
) -> Vec<TextFragment> {
    if special_map.is_empty() || text.is_empty() {
        return vec![TextFragment::Text(text.to_string())];
    }

    // Sort special tokens by length descending (longest match first)
    let mut special_tokens: Vec<_> = special_map.iter().collect();
    special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

    let mut result = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        // Try to match any special token at the start
        let mut found = false;
        for (token_str, &token_id) in &special_tokens {
            if remaining.starts_with(token_str.as_str()) {
                result.push(TextFragment::Special(token_id));
                remaining = &remaining[token_str.len()..];
                found = true;
                break;
            }
        }

        if !found {
            // No special token at start - find the next special token
            let mut next_special_pos = remaining.len();
            for (token_str, _) in &special_tokens {
                if let Some(pos) = remaining.find(token_str.as_str()) {
                    if pos < next_special_pos {
                        next_special_pos = pos;
                    }
                }
            }

            // Add text up to the next special token (or end)
            let text_chunk = &remaining[..next_special_pos];
            if !text_chunk.is_empty() {
                result.push(TextFragment::Text(text_chunk.to_string()));
            }
            remaining = &remaining[next_special_pos..];
        }
    }

    result
}

/// Apply llama.cpp clean_spaces post-processing to decoded text
///
/// This implements the cleanup passes from llama.cpp's detokenize function:
/// 1. Remove space before punctuation: ` ?` → `?`, ` !` → `!`, ` .` → `.`, ` ,` → `,`
/// 2. Strip single apostrophe between spaces: ` ' ` → `'`
/// 3. Merge apostrophe contractions: ` 'm` → `'m`, ` 's` → `'s`, ` 've` → `'ve`, ` 're` → `'re`
fn apply_clean_spaces(text: &str) -> String {
    let mut chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return String::new();
    }

    // Pass 1: Remove space before punctuation ?!.,
    let mut i = 1;
    while i < chars.len() {
        if chars[i - 1] == ' ' && matches!(chars[i], '?' | '!' | '.' | ',') {
            chars.remove(i - 1);
            // Don't increment i since we removed an element
        } else {
            i += 1;
        }
    }

    // Pass 2: Strip single apostrophe between spaces: " ' " → "'"
    let mut i = 1;
    while i + 1 < chars.len() {
        if chars[i] == '\'' && chars[i - 1] == ' ' && chars.get(i + 1) == Some(&' ') {
            // Remove the space before the apostrophe
            chars.remove(i - 1);
            // Now apostrophe is at i-1, space is at i
            chars.remove(i);
            // Don't increment since we modified the array
        } else {
            i += 1;
        }
    }

    // Pass 3: Apostrophe contractions - remove space before certain patterns
    // ` 'm` → `'m`, ` 's` → `'s`, ` 've` → `'ve`, ` 're` → `'re`
    let mut i = 1;
    while i + 1 < chars.len() {
        if chars[i - 1] == ' ' && chars[i] == '\'' {
            let next = chars.get(i + 1);
            let next2 = chars.get(i + 2);

            let should_remove_space = match (next, next2) {
                // ` 's`, ` 'm`
                (Some('s'), _) | (Some('m'), _) => true,
                // ` 've`, ` 're`
                (Some('v'), Some('e')) | (Some('r'), Some('e')) => true,
                // ` 't`, ` 'd`, ` 'll` - llama.cpp comments these out but we include for completeness
                // (Some('t'), _) | (Some('d'), _) => true,
                // (Some('l'), Some('l')) => true,
                _ => false,
            };

            if should_remove_space {
                chars.remove(i - 1);
                // Don't increment
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }

    chars.into_iter().collect()
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

    #[error("Invalid UTF-8: {0}")]
    InvalidUtf8(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
