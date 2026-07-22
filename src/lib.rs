//! # shimmytok
//!
//! Pure Rust tokenizer for GGUF models with 100% llama.cpp compatibility.
//!
//! ## Features
//!
//! - 🦀 **Pure Rust** — No C++ dependencies, compiles anywhere
//! - 📦 **Load from GGUF** — Tokenizer embedded in model file
//! - ✅ **Validated** — 10/10 vocab models match llama.cpp exactly
//! - ⚡ **Fast** — Batch encoding with Rayon parallelism
//! - 🌊 **Streaming** — Token-by-token decoding for LLM output
//!
//! ## Supported Tokenizers
//!
//! | Type | Algorithm | Models |
//! |------|-----------|--------|
//! | SPM | SentencePiece | LLaMA, Mistral, Gemma |
//! | BPE | Byte-Pair Encoding | GPT-2, Qwen, StarCoder, DeepSeek |
//! | WPM | WordPiece | BERT, BGE embeddings |
//! | UGM | Unigram | T5, mT5 |
//! | RWKV | Trie-based | RWKV World |
//! | PLaMo-2 | Table-driven DP | PLaMo-2 |
//!
//! ## Quick Start
//!
//! ```no_run
//! use shimmytok::Tokenizer;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load tokenizer from any GGUF model
//! let tokenizer = Tokenizer::from_gguf_file("model.gguf")?;
//!
//! // Encode text to tokens
//! let tokens = tokenizer.encode("Hello, world!", true)?;
//!
//! // Decode back to text
//! let text = tokenizer.decode(&tokens, true)?;
//!
//! // Stream tokens one at a time (for LLM generation)
//! for token_id in &tokens {
//!     print!("{}", tokenizer.decode_single(*token_id, false)?);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Batch Encoding
//!
//! ```no_run
//! # use shimmytok::Tokenizer;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let tokenizer = Tokenizer::from_gguf_file("model.gguf")?;
//! // Parallel encoding with Rayon
//! let texts = vec!["Hello", "World", "Rust"];
//! let batched = tokenizer.encode_batch(&texts, true)?;
//! # Ok(())
//! # }
//! ```

use rayon::prelude::*;
use std::io::{Cursor, Read};
use std::path::Path;

pub mod bpe;
pub mod byte_encoder;
pub mod gguf;
pub mod invariants;
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

/// Maximum input text size in bytes (10 MB). Encoding larger inputs returns
/// [`Error::TokenizationFailed`].
pub const MAX_INPUT_SIZE: usize = 10 * 1024 * 1024;

/// Maximum number of output tokens (1 M). Prevents unbounded memory use on
/// adversarial or degenerate inputs.
pub const MAX_OUTPUT_TOKENS: usize = 1_000_000;

/// Options for encoding text (llama.cpp parity)
///
/// Construct with [`EncodeOptions::with_special_tokens`] for the common case or
/// [`EncodeOptions::with_parse_special`] when the input may contain literal
/// special-token strings such as `<|eot_id|>`.
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
///
/// Construct with [`DecodeOptions::with_skip_special`] for the common case, or
/// [`DecodeOptions::new`] for full control over whitespace stripping and
/// special-token text emission.
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

/// Type alias for token IDs.
///
/// `u32` matches the GGUF wire format and llama.cpp's internal representation.
/// It is safe because vocabulary size is bounded by [`vocab::Vocabulary`]'s
/// 1 M-token limit, well below `u32::MAX` (4.2 B).
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

/// Generate a `TokenizerImpl` wrapper struct for tokenizers whose `encode`/`decode`
/// methods match the trait signature. This avoids repeating the same forwarding
/// boilerplate for every algorithm variant.
macro_rules! impl_tokenizer_wrapper {
    ($wrapper:ident, $inner:path) => {
        struct $wrapper {
            inner: $inner,
        }

        impl TokenizerImpl for $wrapper {
            fn encode(&self, text: &str, vocab: &Vocabulary) -> Result<Vec<TokenId>, Error> {
                self.inner.encode(text, vocab)
            }
            fn decode(&self, tokens: &[TokenId], vocab: &Vocabulary) -> Result<String, Error> {
                self.inner.decode(tokens, vocab)
            }
        }
    };
}

impl_tokenizer_wrapper!(WpmWrapper, wpm::WpmTokenizer);
impl_tokenizer_wrapper!(RwkvWrapper, rwkv::RwkvTokenizer);
impl_tokenizer_wrapper!(UgmWrapper, ugm::UgmTokenizer);
impl_tokenizer_wrapper!(Plamo2Wrapper, plamo2::Plamo2Tokenizer);

impl Tokenizer {
    /// Load a tokenizer from a GGUF model file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the GGUF file containing model and tokenizer data
    ///
    /// # Errors
    ///
    /// Returns `Err(Error)` if the file cannot be read, is not a valid GGUF
    /// file, or contains an unsupported tokenizer type.
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
        Self::from_vocab(vocab)
    }

    /// Load a tokenizer from any [`Read`] source.
    ///
    /// Useful for loading from network streams, embedded assets, or any
    /// in-memory byte source. For loading from a file path prefer
    /// [`from_gguf_file`](Self::from_gguf_file).
    ///
    /// # Errors
    ///
    /// Same as [`from_gguf_file`](Self::from_gguf_file).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use shimmytok::Tokenizer;
    /// use std::fs::File;
    /// use std::io::BufReader;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let reader = BufReader::new(File::open("model.gguf")?);
    /// let tokenizer = Tokenizer::from_reader(reader)?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "from_reader returns a Result that must be handled"]
    pub fn from_reader<R: Read>(reader: R) -> Result<Self, Error> {
        let vocab = Vocabulary::from_reader(reader)?;
        Self::from_vocab(vocab)
    }

    /// Load a tokenizer from a byte slice containing a GGUF file.
    ///
    /// Convenience wrapper around [`from_reader`](Self::from_reader) for the
    /// common case where the model is already loaded into memory — for example
    /// in WASM applications or embedded systems.
    ///
    /// # Errors
    ///
    /// Same as [`from_gguf_file`](Self::from_gguf_file).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use shimmytok::Tokenizer;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let bytes: Vec<u8> = std::fs::read("model.gguf")?;
    /// let tokenizer = Tokenizer::from_bytes(&bytes)?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "from_bytes returns a Result that must be handled"]
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        Self::from_reader(Cursor::new(bytes))
    }

    /// Shared construction logic — builds a `Tokenizer` from an already-loaded
    /// `Vocabulary`. All public constructors funnel through here.
    fn from_vocab(vocab: Vocabulary) -> Result<Self, Error> {
        let tokenizer_impl: Box<dyn TokenizerImpl> = match vocab.model_type() {
            // SentencePiece models
            "llama" | "mistral" | "gemma" => Box::new(sentencepiece::SentencePieceTokenizer::new()),
            // BPE models
            "gpt2" | "qwen" | "qwen2" => Box::new(bpe::BPETokenizer::new()),
            // WPM (WordPiece) models — BERT-style
            "bert" | "wpm" => Box::new(WpmWrapper {
                inner: wpm::WpmTokenizer::new(&vocab),
            }),
            // RWKV models — trie-based greedy
            "rwkv" => Box::new(RwkvWrapper {
                inner: rwkv::RwkvTokenizer::new(&vocab),
            }),
            // UGM (Unigram) models — T5-style Viterbi
            "t5" | "ugm" => Box::new(UgmWrapper {
                inner: ugm::UgmTokenizer::new(&vocab),
            }),
            // PLaMo-2 models — table-driven DP
            "plamo2" => Box::new(Plamo2Wrapper {
                inner: plamo2::Plamo2Tokenizer::new(&vocab)?,
            }),
            model => return Err(Error::UnsupportedModel(model.to_string())),
        };

        let tokenizer = Self {
            vocab,
            tokenizer_impl,
        };

        // Verify vocabulary consistency in debug builds
        invariants::assert_vocabulary_consistent(&tokenizer);

        Ok(tokenizer)
    }

    /// Get the raw Jinja2 chat template string embedded in the GGUF file.
    ///
    /// Most GGUF models include a `tokenizer.chat_template` field — a Jinja2
    /// template that formats a conversation into the exact prompt format the
    /// model was trained on.
    ///
    /// Pass the returned string to a Jinja renderer such as
    /// [`shimmyjinja`](https://crates.io/crates/shimmyjinja) to produce a
    /// correctly formatted prompt, then encode the result with
    /// [`encode`](Self::encode).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use shimmytok::Tokenizer;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let tokenizer = Tokenizer::from_gguf_file("model.gguf")?;
    ///
    /// if let Some(template) = tokenizer.chat_template() {
    ///     // Pass `template` to shimmyjinja::render_chat_template(...)
    ///     println!("Chat template: {} chars", template.len());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn chat_template(&self) -> Option<&str> {
        self.vocab.chat_template()
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

        // Verify postconditions in debug builds
        invariants::assert_encode_postconditions(&tokens, self.vocab_size());

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
        // NOTE: We intentionally do NOT assert preconditions here because
        // tokens are user input that may be invalid. The code below handles
        // invalid tokens by returning Error::InvalidToken.

        // Filter tokens based on options
        let filtered: Vec<TokenId>;
        let filtered_tokens: &[TokenId] = if options.skip_special_tokens {
            filtered = tokens
                .iter()
                .copied()
                .filter(|&id| !self.vocab.is_special_token(id))
                .collect();
            &filtered
        } else {
            tokens
        };

        // If we need special handling (lstrip or include_special_text=false),
        // we need to decode token by token
        let mut result = if options.lstrip || !options.include_special_text {
            let mut result = String::new();
            for &token_id in filtered_tokens {
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
            self.tokenizer_impl.decode(filtered_tokens, &self.vocab)?
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
    special_tokens.sort_by_key(|b| std::cmp::Reverse(b.0.len()));

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

/// Apply llama.cpp `clean_spaces` post-processing to decoded text.
///
/// Implements the three cleanup passes from llama.cpp's `detokenize`:
///
/// | Pass | Transformation |
/// |------|---------------|
/// | 1 | Remove space before `?`, `!`, `.`, `,` |
/// | 2 | Strip isolated apostrophe surrounded by spaces: ` ' ` → `'` |
/// | 3 | Merge contractions: ` 'm`, ` 's`, ` 've`, ` 're` → drop the space |
///
/// The implementation collects to `Vec<char>` so that O(1) index access
/// across all three passes avoids repeated UTF-8 scanning.
fn apply_clean_spaces(text: &str) -> String {
    let mut chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return String::new();
    }

    // Pass 1: remove the space immediately before ?  !  .  ,
    let mut i = 1;
    while i < chars.len() {
        if chars[i - 1] == ' ' && matches!(chars[i], '?' | '!' | '.' | ',') {
            chars.remove(i - 1);
            // Don't advance — the element at i-1 is now what was at i+1.
        } else {
            i += 1;
        }
    }

    // Pass 2: collapse ` ' ` (apostrophe flanked by spaces) to just `'`
    let mut i = 1;
    while i + 1 < chars.len() {
        if chars[i] == '\'' && chars[i - 1] == ' ' && chars[i + 1] == ' ' {
            chars.remove(i - 1); // remove leading space
            chars.remove(i); // remove trailing space (now at index i)
        } else {
            i += 1;
        }
    }

    // Pass 3: remove the space before English contractions
    //   ` 'm`, ` 's`, ` 've`, ` 're`
    let mut i = 1;
    while i + 1 < chars.len() {
        if chars[i - 1] == ' ' && chars[i] == '\'' {
            let next = chars.get(i + 1);
            let next2 = chars.get(i + 2);

            let is_contraction = matches!(
                (next, next2),
                (Some('s' | 'm'), _) | (Some('v' | 'r'), Some('e'))
            );

            if is_contraction {
                chars.remove(i - 1);
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }

    chars.into_iter().collect()
}

/// Errors that can be returned by shimmytok operations.
///
/// This enum is `#[non_exhaustive]` — new variants may be added in minor
/// releases without breaking downstream `match` expressions that include a
/// `_` arm.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
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
