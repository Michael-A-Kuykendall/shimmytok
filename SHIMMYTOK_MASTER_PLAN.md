# ShimmyTok Master Plan
**Created**: 2025-10-20  
**Goal**: Pure Rust GGUF tokenization library (FIRST IN RUST ECOSYSTEM)  
**Status**: Architecture phase - ready to implement

---

## Executive Summary

### What We're Building
A standalone Rust crate that reads GGUF model files and provides tokenization/detokenization using the vocabulary embedded in the file.

### Why This Matters
- **35+ GGUF parsing crates exist in Rust**
- **ZERO implement tokenization**
- **Gap exists for 3+ years** (oldest crates from 2022)
- **Estimated 50-100+ potential users** (every Rust GGUF project needs tokenization)

### What Makes This Different
We're **NOT porting llama.cpp** - we're implementing **published algorithms**:
- SentencePiece: Google's 2018 paper
- BPE: Sennrich et al. 2015 paper
- Unicode: Standard normalization

The ONLY llama.cpp-specific part is reading GGUF vocab format (reverse-engineerable).

---

## The Opportunity (Why NOW)

### Market Research: Existing Rust GGUF Crates
Searched `crates.io` on 2025-10-20:

```
gguf = "0.1.2"                  # Parser with CLI tool
gguf-rs-lib = "0.2.5"          # Read/write GGUF files
woolly-gguf = "0.0.0"          # Zero-copy loader
inspector-gguf = "0.3.0"       # Inspection tool
gguf-utils = "0.1.0"           # Utilities
... and 30+ more
```

**ALL parse GGUF** → **ZERO do tokenization**

### Why the Gap Exists
1. **Complexity**: llama.cpp tokenization is ~2000 LOC
2. **Niche**: Seems specific to llama.cpp
3. **Moving target**: Models change frequently
4. **Perceived as port**: People think you MUST port llama.cpp

### Why We Can Succeed
1. ✅ **Algorithms are public** (not llama.cpp secrets)
2. ✅ **We can validate** (test against llama.cpp)
3. ✅ **Clean slate** (no legacy baggage)
4. ✅ **Rust advantages** (safety, ecosystem)
5. ✅ **First mover** (no competition)

---

## Core Architecture

### Three-Layer Design

```
┌─────────────────────────────────────────────────────┐
│                  shimmytok                          │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Layer 1: GGUF Vocab Loader (llama.cpp-specific)  │
│  ┌─────────────────────────────────────────────┐  │
│  │ • Read tokenizer.ggml.* metadata            │  │
│  │ • Parse token strings, scores, types        │  │
│  │ • Extract special tokens (BOS/EOS/UNK)      │  │
│  │ • Build bidirectional token↔ID mapping      │  │
│  └─────────────────────────────────────────────┘  │
│                      ↓                             │
│  Layer 2: Algorithm Implementations (published)    │
│  ┌─────────────────────────────────────────────┐  │
│  │ • SentencePiece (Google 2018 paper)         │  │
│  │ • BPE (Sennrich 2015 paper)                 │  │
│  │ • Unicode normalization (crate)             │  │
│  │ • Viterbi DP (textbook algorithm)           │  │
│  └─────────────────────────────────────────────┘  │
│                      ↓                             │
│  Layer 3: High-Level API (Rusty interface)        │
│  ┌─────────────────────────────────────────────┐  │
│  │ • Tokenizer::from_gguf()                    │  │
│  │ • encode(text) -> Vec<TokenId>              │  │
│  │ • decode(tokens) -> String                  │  │
│  │ • Special token handling                    │  │
│  └─────────────────────────────────────────────┘  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Implementation Details

### File Structure

```
shimmytok/
├── Cargo.toml
├── README.md
├── SHIMMYTOK_MASTER_PLAN.md        # This file
├── LICENSE                          # MIT or Apache-2.0
├── src/
│   ├── lib.rs                      # Public API + re-exports
│   ├── vocab.rs                    # GGUF vocab loading (Layer 1)
│   ├── sentencepiece.rs            # SentencePiece algorithm (Layer 2)
│   ├── bpe.rs                      # BPE algorithm (Layer 2)
│   ├── normalize.rs                # Unicode normalization (Layer 2)
│   ├── special_tokens.rs           # BOS/EOS/UNK handling
│   ├── error.rs                    # Error types
│   └── gguf_metadata.rs            # GGUF metadata reader (minimal)
├── tests/
│   ├── test_vocab_loading.rs      # Layer 1 tests
│   ├── test_sentencepiece.rs      # Layer 2 tests
│   ├── test_bpe.rs                 # Layer 2 tests
│   ├── test_roundtrip.rs           # Encode→decode tests
│   ├── test_against_llama.rs      # Validation tests
│   └── fixtures/
│       ├── tinyllama.gguf          # Test model
│       └── expected_tokens.json    # Known-good outputs
├── examples/
│   ├── basic_usage.rs              # Simple encode/decode
│   ├── compare_with_llama.rs       # Show we match llama.cpp
│   ├── inspect_vocab.rs            # Explore GGUF vocab
│   └── benchmark.rs                # Performance testing
└── benches/
    └── tokenization.rs             # Criterion benchmarks
```

---

## Public API Design

### Primary Interface

```rust
// lib.rs - Clean, Rusty API

use std::path::Path;

pub type TokenId = u32;

/// Main tokenizer interface
pub struct Tokenizer {
    vocab: Vocabulary,
    algorithm: Algorithm,
}

enum Algorithm {
    SentencePiece(SentencePieceTokenizer),
    BPE(BPETokenizer),
}

impl Tokenizer {
    /// Load tokenizer from GGUF file
    /// 
    /// # Example
    /// ```
    /// let tok = Tokenizer::from_gguf_file("model.gguf")?;
    /// let tokens = tok.encode("Hello world", false)?;
    /// ```
    pub fn from_gguf_file<P: AsRef<Path>>(path: P) -> Result<Self, Error>;
    
    /// Load from already-parsed GGUF metadata
    /// (for users who already have GGUF parser)
    pub fn from_gguf_metadata(metadata: &GGUFMetadata) -> Result<Self, Error>;
    
    /// Encode text to token IDs
    /// 
    /// # Arguments
    /// * `text` - Text to tokenize
    /// * `add_special_tokens` - Whether to add BOS/EOS tokens
    /// 
    /// # Returns
    /// Vector of token IDs
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<TokenId>, Error>;
    
    /// Decode token IDs back to text
    /// 
    /// # Arguments
    /// * `tokens` - Token IDs to decode
    /// * `skip_special_tokens` - Whether to skip BOS/EOS/etc in output
    /// 
    /// # Returns
    /// Decoded text string
    pub fn decode(&self, tokens: &[TokenId], skip_special_tokens: bool) -> Result<String, Error>;
    
    /// Encode text and return both tokens and their string representations
    /// (useful for debugging)
    pub fn encode_with_pieces(&self, text: &str) -> Result<Vec<(TokenId, String)>, Error>;
    
    /// Get special token IDs
    pub fn bos_token_id(&self) -> TokenId;
    pub fn eos_token_id(&self) -> TokenId;
    pub fn unk_token_id(&self) -> TokenId;
    pub fn pad_token_id(&self) -> Option<TokenId>;
    
    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize;
    
    /// Get tokenizer type (for introspection)
    pub fn tokenizer_type(&self) -> TokenizerType;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerType {
    SentencePiece,
    BPE,
}

/// Error types for shimmytok
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Failed to read GGUF file: {0}")]
    GGUFRead(String),
    
    #[error("Invalid GGUF metadata: {0}")]
    InvalidMetadata(String),
    
    #[error("Unsupported tokenizer type: {0}")]
    UnsupportedTokenizer(String),
    
    #[error("Tokenization failed: {0}")]
    TokenizationFailed(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
```

### Advanced API (Power Users)

```rust
/// Direct access to vocabulary (for advanced users)
pub struct Vocabulary {
    tokens: Vec<String>,
    token_to_id: HashMap<String, TokenId>,
    scores: Vec<f32>,
    token_types: Vec<TokenType>,
    special_tokens: SpecialTokens,
}

impl Vocabulary {
    /// Get token string by ID
    pub fn get_token(&self, id: TokenId) -> Option<&str>;
    
    /// Get token ID by string
    pub fn get_id(&self, token: &str) -> Option<TokenId>;
    
    /// Get token score (for SentencePiece)
    pub fn score(&self, id: TokenId) -> f32;
    
    /// Get token type
    pub fn token_type(&self, id: TokenId) -> TokenType;
    
    /// Iterate over all tokens
    pub fn iter(&self) -> impl Iterator<Item = (TokenId, &str)>;
    
    /// Vocabulary size
    pub fn len(&self) -> usize;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    Normal = 1,      // Regular token
    Unknown = 2,     // UNK token
    Control = 3,     // BOS/EOS/etc
    UserDefined = 4, // Custom tokens
    Unused = 5,      // Placeholder
    Byte = 6,        // Byte fallback
}

#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub bos: TokenId,
    pub eos: TokenId,
    pub unk: TokenId,
    pub pad: Option<TokenId>,
}
```

---

## Algorithm Implementations

### 1. SentencePiece (Google 2018)

**Source**: "SentencePiece: A simple and language independent approach to subword tokenization" (Kudo & Richardson, 2018)

**Algorithm**: Viterbi + Dynamic Programming

```rust
// sentencepiece.rs

/// SentencePiece Unigram tokenizer
/// 
/// Implementation based on Google's published paper, NOT llama.cpp port
/// Paper: https://arxiv.org/abs/1808.06226
pub struct SentencePieceTokenizer {
    vocab: Vocabulary,
}

impl SentencePieceTokenizer {
    pub fn new(vocab: Vocabulary) -> Self {
        Self { vocab }
    }
    
    /// Tokenize using Viterbi algorithm
    /// 
    /// This is standard dynamic programming - well known in NLP
    /// NOT specific to llama.cpp
    pub fn encode(&self, text: &str) -> Vec<TokenId> {
        // Step 1: Normalize Unicode (NFC normalization)
        let normalized = normalize_nfc(text);
        
        // Step 2: Build lattice + find best path (Viterbi)
        let n = normalized.len();
        let mut best_score = vec![f32::NEG_INFINITY; n + 1];
        let mut best_prev: Vec<Option<(TokenId, usize)>> = vec![None; n + 1];
        best_score[0] = 0.0;
        
        // For each position in text
        for i in 0..n {
            if best_score[i].is_infinite() {
                continue; // Unreachable position
            }
            
            // Try all vocab tokens that could start at position i
            for (token_str, &token_id) in self.vocab.token_to_id.iter() {
                if normalized[i..].starts_with(token_str) {
                    let next_pos = i + token_str.len();
                    let score = best_score[i] + self.vocab.score(token_id);
                    
                    // Update if better path found
                    if score > best_score[next_pos] {
                        best_score[next_pos] = score;
                        best_prev[next_pos] = Some((token_id, i));
                    }
                }
            }
        }
        
        // Step 3: Backtrack to get token sequence
        let mut tokens = Vec::new();
        let mut pos = n;
        
        while pos > 0 {
            match best_prev[pos] {
                Some((token_id, prev_pos)) => {
                    tokens.push(token_id);
                    pos = prev_pos;
                }
                None => {
                    // Fallback: use UNK token for unmatchable character
                    tokens.push(self.vocab.special_tokens.unk);
                    pos = pos.saturating_sub(1);
                }
            }
        }
        
        tokens.reverse();
        tokens
    }
    
    pub fn decode(&self, tokens: &[TokenId]) -> String {
        tokens.iter()
            .filter_map(|&id| self.vocab.get_token(id))
            .collect::<Vec<_>>()
            .join("")
    }
}
```

**Complexity**: O(n * V) where n = text length, V = vocab size
**Optimization opportunity**: Trie for faster vocab lookup (O(n * log V))

### 2. BPE (Sennrich et al. 2015)

**Source**: "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2015)

**Algorithm**: Greedy merge

```rust
// bpe.rs

/// Byte-Pair Encoding tokenizer
/// 
/// Implementation based on Sennrich et al. paper, NOT llama.cpp port
/// Paper: https://arxiv.org/abs/1508.07909
pub struct BPETokenizer {
    vocab: Vocabulary,
    merges: Vec<(String, String)>, // Learned merge rules
}

impl BPETokenizer {
    pub fn new(vocab: Vocabulary) -> Self {
        // Extract merge rules from vocabulary
        // In GGUF, merges are implicit in token ordering
        let merges = Self::extract_merges(&vocab);
        
        Self { vocab, merges }
    }
    
    fn extract_merges(vocab: &Vocabulary) -> Vec<(String, String)> {
        // BPE merges are encoded in the vocabulary structure
        // Tokens created by merging appear later in vocab
        // Need to reconstruct merge sequence from vocab
        
        let mut merges = Vec::new();
        
        // For each token, check if it's a merge of two earlier tokens
        for (id, token) in vocab.iter() {
            // Try to split token into two parts that exist in vocab
            for split_pos in 1..token.len() {
                let left = &token[..split_pos];
                let right = &token[split_pos..];
                
                if let (Some(left_id), Some(right_id)) = (vocab.get_id(left), vocab.get_id(right)) {
                    // Both parts exist and come before this token
                    if left_id < id && right_id < id {
                        merges.push((left.to_string(), right.to_string()));
                        break;
                    }
                }
            }
        }
        
        merges
    }
    
    pub fn encode(&self, text: &str) -> Vec<TokenId> {
        // Step 1: Normalize
        let normalized = normalize_nfc(text);
        
        // Step 2: Split into initial tokens (characters or bytes)
        let mut tokens: Vec<String> = normalized
            .chars()
            .map(|c| c.to_string())
            .collect();
        
        // Step 3: Apply merges in order
        for (left, right) in &self.merges {
            let merged = format!("{}{}", left, right);
            
            // Find and merge all occurrences of this pair
            let mut i = 0;
            while i + 1 < tokens.len() {
                if tokens[i] == *left && tokens[i + 1] == *right {
                    tokens[i] = merged.clone();
                    tokens.remove(i + 1);
                }
                i += 1;
            }
        }
        
        // Step 4: Convert token strings to IDs
        tokens.iter()
            .map(|t| self.vocab.get_id(t).unwrap_or(self.vocab.special_tokens.unk))
            .collect()
    }
    
    pub fn decode(&self, tokens: &[TokenId]) -> String {
        tokens.iter()
            .filter_map(|&id| self.vocab.get_token(id))
            .collect::<Vec<_>>()
            .join("")
    }
}
```

**Complexity**: O(n * M) where n = text length, M = number of merges
**Optimization opportunity**: Priority queue for merge selection

### 3. Unicode Normalization

**Source**: Unicode Standard (UAX #15)

```rust
// normalize.rs

use unicode_normalization::UnicodeNormalization;

/// Normalize text using NFC (Canonical Composition)
/// 
/// This is standard Unicode normalization per UAX #15
/// Using battle-tested unicode-normalization crate
pub fn normalize_nfc(text: &str) -> String {
    text.nfc().collect()
}

/// Normalize text using NFD (Canonical Decomposition)
pub fn normalize_nfd(text: &str) -> String {
    text.nfd().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_nfc_normalization() {
        // é can be: U+00E9 (precomposed) or U+0065 U+0301 (decomposed)
        let decomposed = "e\u{0301}";
        let normalized = normalize_nfc(decomposed);
        assert_eq!(normalized, "é");
    }
    
    #[test]
    fn test_emoji() {
        // Emoji with skin tone modifier
        let emoji = "👍🏽"; // Thumbs up + medium skin tone
        let normalized = normalize_nfc(emoji);
        assert_eq!(normalized, emoji); // Should preserve
    }
}
```

---

## GGUF Vocab Format (The llama.cpp Part)

### Metadata Keys

Based on inspection of TinyLlama GGUF file:

```
tokenizer.ggml.model = "llama"  # or "gpt2", "bert", etc.
tokenizer.ggml.tokens = [...]   # Array of token strings
tokenizer.ggml.scores = [...]   # Array of f32 scores (SentencePiece)
tokenizer.ggml.token_type = [...] # Array of i32 types
tokenizer.ggml.bos_token_id = 1
tokenizer.ggml.eos_token_id = 2
tokenizer.ggml.unknown_token_id = 0
tokenizer.ggml.padding_token_id = ? # Optional
tokenizer.ggml.merges = [...]   # BPE merges (if BPE model)
```

### Vocab Loader Implementation

```rust
// vocab.rs

use std::collections::HashMap;
use std::path::Path;

/// Vocabulary loaded from GGUF metadata
pub struct Vocabulary {
    tokens: Vec<String>,
    token_to_id: HashMap<String, TokenId>,
    scores: Vec<f32>,
    token_types: Vec<TokenType>,
    pub special_tokens: SpecialTokens,
}

impl Vocabulary {
    /// Load vocabulary from GGUF file
    pub fn from_gguf_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let metadata = GGUFMetadata::read_file(path)?;
        Self::from_metadata(&metadata)
    }
    
    /// Load from parsed metadata
    pub fn from_metadata(meta: &GGUFMetadata) -> Result<Self, Error> {
        // Read token strings
        let tokens = meta.get_string_array("tokenizer.ggml.tokens")
            .ok_or_else(|| Error::InvalidMetadata("Missing tokenizer.ggml.tokens".into()))?;
        
        // Read scores (for SentencePiece)
        let scores = meta.get_f32_array("tokenizer.ggml.scores")
            .unwrap_or_else(|| vec![0.0; tokens.len()]);
        
        // Read token types
        let token_types = meta.get_i32_array("tokenizer.ggml.token_type")
            .unwrap_or_else(|| vec![TokenType::Normal as i32; tokens.len()])
            .into_iter()
            .map(TokenType::from_i32)
            .collect();
        
        // Build token → ID mapping
        let token_to_id: HashMap<String, TokenId> = tokens.iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id as TokenId))
            .collect();
        
        // Read special tokens
        let special_tokens = SpecialTokens {
            bos: meta.get_u32("tokenizer.ggml.bos_token_id")
                .ok_or_else(|| Error::InvalidMetadata("Missing BOS token".into()))?,
            eos: meta.get_u32("tokenizer.ggml.eos_token_id")
                .ok_or_else(|| Error::InvalidMetadata("Missing EOS token".into()))?,
            unk: meta.get_u32("tokenizer.ggml.unknown_token_id")
                .ok_or_else(|| Error::InvalidMetadata("Missing UNK token".into()))?,
            pad: meta.get_u32("tokenizer.ggml.padding_token_id").ok(),
        };
        
        Ok(Vocabulary {
            tokens,
            token_to_id,
            scores,
            token_types,
            special_tokens,
        })
    }
    
    pub fn get_token(&self, id: TokenId) -> Option<&str> {
        self.tokens.get(id as usize).map(|s| s.as_str())
    }
    
    pub fn get_id(&self, token: &str) -> Option<TokenId> {
        self.token_to_id.get(token).copied()
    }
    
    pub fn score(&self, id: TokenId) -> f32 {
        self.scores.get(id as usize).copied().unwrap_or(0.0)
    }
    
    pub fn token_type(&self, id: TokenId) -> TokenType {
        self.token_types.get(id as usize).copied().unwrap_or(TokenType::Normal)
    }
    
    pub fn len(&self) -> usize {
        self.tokens.len()
    }
    
    pub fn iter(&self) -> impl Iterator<Item = (TokenId, &str)> {
        self.tokens.iter().enumerate().map(|(id, s)| (id as TokenId, s.as_str()))
    }
}

impl TokenType {
    fn from_i32(value: i32) -> Self {
        match value {
            1 => TokenType::Normal,
            2 => TokenType::Unknown,
            3 => TokenType::Control,
            4 => TokenType::UserDefined,
            5 => TokenType::Unused,
            6 => TokenType::Byte,
            _ => TokenType::Normal, // Default fallback
        }
    }
}
```

### Minimal GGUF Metadata Reader

```rust
// gguf_metadata.rs - Minimal reader, just enough for vocab

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// Minimal GGUF metadata reader
/// 
/// We only need to read metadata key-value pairs for tokenizer config
/// Don't need to handle tensors (that's for model loading)
pub struct GGUFMetadata {
    kv_pairs: HashMap<String, MetadataValue>,
}

#[derive(Debug, Clone)]
pub enum MetadataValue {
    U32(u32),
    I32(i32),
    F32(f32),
    String(String),
    StringArray(Vec<String>),
    F32Array(Vec<f32>),
    I32Array(Vec<i32>),
}

impl GGUFMetadata {
    pub fn read_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        
        // Read GGUF header
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        
        if &magic != b"GGUF" {
            return Err(Error::InvalidMetadata("Not a GGUF file".into()));
        }
        
        // Read version
        let version = read_u32(&mut reader)?;
        
        // Read tensor count and metadata count
        let tensor_count = read_u64(&mut reader)?;
        let metadata_count = read_u64(&mut reader)?;
        
        // Read metadata key-value pairs
        let mut kv_pairs = HashMap::new();
        
        for _ in 0..metadata_count {
            let key = read_string(&mut reader)?;
            let value = read_metadata_value(&mut reader)?;
            kv_pairs.insert(key, value);
        }
        
        Ok(GGUFMetadata { kv_pairs })
    }
    
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.kv_pairs.get(key)? {
            MetadataValue::U32(v) => Some(*v),
            _ => None,
        }
    }
    
    pub fn get_string_array(&self, key: &str) -> Option<Vec<String>> {
        match self.kv_pairs.get(key)? {
            MetadataValue::StringArray(v) => Some(v.clone()),
            _ => None,
        }
    }
    
    pub fn get_f32_array(&self, key: &str) -> Option<Vec<f32>> {
        match self.kv_pairs.get(key)? {
            MetadataValue::F32Array(v) => Some(v.clone()),
            _ => None,
        }
    }
    
    pub fn get_i32_array(&self, key: &str) -> Option<Vec<i32>> {
        match self.kv_pairs.get(key)? {
            MetadataValue::I32Array(v) => Some(v.clone()),
            _ => None,
        }
    }
}

// Helper functions for reading GGUF format
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

fn read_string<R: Read>(reader: &mut R) -> Result<String, Error> {
    let len = read_u64(reader)? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| Error::InvalidMetadata(format!("Invalid UTF-8: {}", e)))
}

fn read_metadata_value<R: Read>(reader: &mut R) -> Result<MetadataValue, Error> {
    let type_id = read_u32(reader)?;
    
    match type_id {
        4 => Ok(MetadataValue::U32(read_u32(reader)?)),
        5 => Ok(MetadataValue::I32(read_i32(reader)?)),
        6 => Ok(MetadataValue::F32(read_f32(reader)?)),
        8 => Ok(MetadataValue::String(read_string(reader)?)),
        9 => { // Array
            let array_type = read_u32(reader)?;
            let array_len = read_u64(reader)? as usize;
            
            match array_type {
                8 => { // String array
                    let mut arr = Vec::with_capacity(array_len);
                    for _ in 0..array_len {
                        arr.push(read_string(reader)?);
                    }
                    Ok(MetadataValue::StringArray(arr))
                }
                6 => { // F32 array
                    let mut arr = Vec::with_capacity(array_len);
                    for _ in 0..array_len {
                        arr.push(read_f32(reader)?);
                    }
                    Ok(MetadataValue::F32Array(arr))
                }
                5 => { // I32 array
                    let mut arr = Vec::with_capacity(array_len);
                    for _ in 0..array_len {
                        arr.push(read_i32(reader)?);
                    }
                    Ok(MetadataValue::I32Array(arr))
                }
                _ => Err(Error::InvalidMetadata(format!("Unsupported array type: {}", array_type)))
            }
        }
        _ => Err(Error::InvalidMetadata(format!("Unsupported metadata type: {}", type_id)))
    }
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
```

---

## Dependencies

```toml
[package]
name = "shimmytok"
version = "0.1.0"
edition = "2021"
license = "MIT OR Apache-2.0"
description = "Pure Rust GGUF tokenization library"
repository = "https://github.com/Michael-A-Kuykendall/shimmytok"
keywords = ["tokenizer", "gguf", "llama", "nlp", "sentencepiece"]
categories = ["text-processing", "parsing"]

[dependencies]
# Unicode normalization (battle-tested)
unicode-normalization = "0.1"

# Error handling
thiserror = "1.0"

# Optional: use existing GGUF parser instead of our minimal one
# gguf-rs-lib = { version = "0.2", optional = true }

[dev-dependencies]
# Testing
criterion = "0.5"    # Benchmarks
tempfile = "3.0"     # Test fixtures
serde_json = "1.0"   # Test data serialization

[features]
default = []
# Use external GGUF library (if we want to avoid maintaining our own reader)
external-gguf = ["gguf-rs-lib"]
```

---

## Testing Strategy

### Unit Tests (Per Module)

```rust
// tests/test_vocab_loading.rs

#[test]
fn test_load_tinyllama_vocab() {
    let vocab = Vocabulary::from_gguf_file("fixtures/tinyllama.gguf").unwrap();
    
    // TinyLlama has 32,000 tokens
    assert_eq!(vocab.len(), 32000);
    
    // Check special tokens
    assert_eq!(vocab.special_tokens.bos, 1);
    assert_eq!(vocab.special_tokens.eos, 2);
    
    // Check known tokens
    assert_eq!(vocab.get_id("Hello"), Some(12345)); // Example
}

#[test]
fn test_token_scores() {
    let vocab = Vocabulary::from_gguf_file("fixtures/tinyllama.gguf").unwrap();
    
    // More common tokens should have higher (less negative) scores
    let common_score = vocab.score(vocab.get_id("the").unwrap());
    let rare_score = vocab.score(vocab.get_id("xylophone").unwrap());
    
    assert!(common_score > rare_score);
}
```

```rust
// tests/test_sentencepiece.rs

#[test]
fn test_simple_tokenization() {
    let vocab = Vocabulary::from_gguf_file("fixtures/tinyllama.gguf").unwrap();
    let tok = SentencePieceTokenizer::new(vocab);
    
    let tokens = tok.encode("Hello world");
    
    // Should produce reasonable tokenization
    assert!(!tokens.is_empty());
    assert!(tokens.len() <= 10); // Shouldn't over-segment
}

#[test]
fn test_roundtrip() {
    let vocab = Vocabulary::from_gguf_file("fixtures/tinyllama.gguf").unwrap();
    let tok = SentencePieceTokenizer::new(vocab);
    
    let text = "The quick brown fox jumps over the lazy dog.";
    let tokens = tok.encode(text);
    let decoded = tok.decode(&tokens);
    
    // Should roundtrip (might have minor whitespace differences)
    assert_eq!(text.trim(), decoded.trim());
}
```

### Integration Tests (Against llama.cpp)

```rust
// tests/test_against_llama.rs

/// Compare our tokenization with llama.cpp output
/// 
/// Run llama.cpp separately and save expected outputs to JSON
/// Then validate we match exactly
#[test]
fn test_matches_llama_cpp() {
    let test_cases: Vec<(String, Vec<TokenId>)> = 
        serde_json::from_str(include_str!("fixtures/expected_tokens.json")).unwrap();
    
    let vocab = Vocabulary::from_gguf_file("fixtures/tinyllama.gguf").unwrap();
    let tokenizer = Tokenizer::from_gguf_file("fixtures/tinyllama.gguf").unwrap();
    
    for (text, expected_tokens) in test_cases {
        let our_tokens = tokenizer.encode(&text, false).unwrap();
        
        assert_eq!(
            our_tokens, expected_tokens,
            "Mismatch for text: {:?}", text
        );
    }
}
```

### Fuzzing

```rust
// tests/fuzz_roundtrip.rs

use proptest::prelude::*;

proptest! {
    #[test]
    fn fuzz_roundtrip(text in "\\PC*") {
        let tokenizer = get_test_tokenizer();
        
        // Tokenize and detokenize
        if let Ok(tokens) = tokenizer.encode(&text, false) {
            let decoded = tokenizer.decode(&tokens, false).unwrap();
            
            // Should be similar (Unicode normalization may change things)
            // Use fuzzy comparison
            prop_assert!(similar(&text, &decoded, 0.95));
        }
    }
}
```

---

## Development Plan

### Week 1: GGUF Vocab Loading
**Goal**: Read vocabulary from GGUF files

**Tasks**:
- [ ] Implement minimal GGUF metadata reader (`gguf_metadata.rs`)
- [ ] Implement `Vocabulary::from_gguf_file()`
- [ ] Test with TinyLlama GGUF file
- [ ] Validate:
  - Token count correct
  - Special tokens correct
  - Can access tokens by ID and string

**Deliverable**: Can load and inspect GGUF vocabulary

**Validation**:
```bash
cargo test test_vocab_loading
```

### Week 2: SentencePiece Algorithm
**Goal**: Implement SentencePiece from published paper

**Tasks**:
- [ ] Implement Unicode NFC normalization wrapper
- [ ] Implement Viterbi DP algorithm (`sentencepiece.rs`)
- [ ] Test with simple cases ("Hello world", "The cat sat")
- [ ] Compare with llama.cpp output (run llama-tokenize)
- [ ] Fix any discrepancies

**Deliverable**: SentencePiece working for TinyLlama

**Validation**:
```bash
cargo test test_sentencepiece
cargo run --example basic_usage
```

### Week 3: BPE Algorithm
**Goal**: Implement BPE from published paper

**Tasks**:
- [ ] Figure out BPE merge extraction from GGUF vocab
- [ ] Implement BPE greedy merge algorithm (`bpe.rs`)
- [ ] Test with GPT-2 style model (if we can find GGUF)
- [ ] Compare with llama.cpp
- [ ] Fix discrepancies

**Deliverable**: BPE working for GPT-style models

**Validation**:
```bash
cargo test test_bpe
```

### Week 4: Edge Cases & Validation
**Goal**: Handle all the weird stuff

**Tasks**:
- [ ] Test with multiple models (LLaMA, Phi, etc.)
- [ ] Handle unknown characters properly (byte fallback)
- [ ] Special token edge cases (multiple BOS/EOS, etc.)
- [ ] Unicode edge cases (emoji, combining chars, RTL text)
- [ ] Create comprehensive test suite with expected outputs

**Deliverable**: Robust, tested tokenizer

**Validation**:
```bash
cargo test --all
cargo test test_against_llama
```

### Week 5: Polish & Release
**Goal**: Make it production-ready

**Tasks**:
- [ ] Write comprehensive documentation
- [ ] Create examples (basic_usage, compare_with_llama, benchmark)
- [ ] Performance benchmarks with Criterion
- [ ] Write README with usage examples
- [ ] Add CI/CD (GitHub Actions)
- [ ] Publish to crates.io

**Deliverable**: Released crate v0.1.0

**Validation**:
```bash
cargo doc --open
cargo bench
cargo publish --dry-run
```

---

## Performance Targets

### Baseline (Good Enough)
- **Throughput**: >10,000 tokens/sec (simple text)
- **Latency**: <1ms for typical sentences (20-50 chars)
- **Memory**: <10MB overhead for vocab

### Stretch Goals (If Time Permits)
- **Throughput**: >100,000 tokens/sec (with optimizations)
- **Latency**: <100μs for typical sentences
- **Memory**: <5MB overhead

### Optimization Opportunities
1. **Trie for vocab lookup** - O(n log V) → O(n)
2. **Parallel batch tokenization** - use rayon for multiple texts
3. **Token caching** - cache frequent phrases
4. **SIMD string matching** - fast prefix matching

---

## Known Challenges

### Challenge 1: GGUF Format Variations
**Problem**: Different GGUF versions might have slightly different formats

**Mitigation**:
- Test with multiple model sources (HuggingFace, TheBloke, etc.)
- Graceful fallbacks for missing fields
- Version detection and handling

### Challenge 2: Model-Specific Quirks
**Problem**: Different models might handle edge cases differently

**Mitigation**:
- Extensive testing with diverse models
- Clear documentation of tested models
- Community feedback loop

### Challenge 3: Unicode Edge Cases
**Problem**: Unicode is complex (emoji, RTL, combining chars)

**Mitigation**:
- Use battle-tested `unicode-normalization` crate
- Comprehensive Unicode test suite
- Validate against llama.cpp on edge cases

### Challenge 4: BPE Merge Extraction
**Problem**: GGUF doesn't explicitly store BPE merges

**Mitigation**:
- Reverse-engineer from vocab structure
- Test with known BPE models
- Fall back to asking llama.cpp community if needed

---

## Success Criteria

### Minimum Viable Product (MVP)
- [ ] Loads vocab from TinyLlama GGUF
- [ ] SentencePiece tokenization matches llama.cpp exactly
- [ ] Detokenization works correctly
- [ ] Special tokens handled (BOS/EOS/UNK)
- [ ] Basic tests passing
- [ ] Documented API

### Production Ready (v0.1.0)
- [ ] Works with 3+ model architectures (LLaMA, Phi, GPT)
- [ ] 100+ test cases passing
- [ ] Matches llama.cpp output exactly on test suite
- [ ] Performance meets baseline targets
- [ ] Comprehensive documentation
- [ ] Examples and benchmarks
- [ ] Published to crates.io

### Ecosystem Success (v0.2.0+)
- [ ] Used by 5+ projects
- [ ] Supports 10+ model types
- [ ] Community contributions
- [ ] Performance optimizations (trie, SIMD)
- [ ] Batch tokenization API

---

## Why This Will Succeed

### Technical Reasons
1. ✅ **Algorithms are public** - Not guessing, implementing specs
2. ✅ **We can validate** - Test against llama.cpp at every step
3. ✅ **Manageable scope** - ~1000 LOC, 5 weeks
4. ✅ **Rust advantages** - Safety, zero-cost abstractions, great ecosystem

### Strategic Reasons
1. ✅ **First mover** - No competition in Rust
2. ✅ **Clear need** - 35+ GGUF crates need this
3. ✅ **Standalone value** - Useful independent of libshimmy
4. ✅ **Growing market** - GGUF adoption increasing

### Psychological Reasons
1. ✅ **We've done harder** - Quantization was more complex
2. ✅ **Clean slate** - No legacy baggage
3. ✅ **Auditable** - This doc ensures we stay on track
4. ✅ **Fast feedback** - Can test against llama.cpp immediately

---

## Risks & Mitigation

### Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| GGUF format undocumented | Medium | Medium | Test with real files, ask community |
| Algorithm doesn't match | Low | High | Validate against llama.cpp continuously |
| Model-specific quirks | High | Medium | Test diverse models, clear docs |
| Performance insufficient | Low | Low | Profile early, optimize later |
| Unicode edge cases | Medium | Low | Use proven crate, comprehensive tests |
| BPE merge extraction | Medium | Medium | Study vocab structure, community help |

### Contingency Plans

**If GGUF format too complex**:
→ Use gguf-rs-lib crate instead of custom reader

**If algorithms don't match llama.cpp**:
→ Identify specific mismatches, port those edge case handlers

**If performance insufficient**:
→ Profile, optimize hot paths (trie, SIMD)

**If BPE merges unsolvable**:
→ Start with SentencePiece only, add BPE later

---

## Integration with libshimmy

### Current State
libshimmy needs tokenization for Phase 1 Task 2 (8 Fibonacci points)

### Integration Plan

```toml
# libshimmy/Cargo.toml
[dependencies]
shimmytok = { path = "../shimmytok" }
```

```rust
// libshimmy/src/text_generation.rs

use shimmytok::Tokenizer;

pub struct TextGenerator {
    model: Model,
    tokenizer: Tokenizer,
}

impl TextGenerator {
    pub fn from_gguf_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let model = Model::from_gguf_file(&path)?;
        let tokenizer = Tokenizer::from_gguf_file(&path)?;
        
        Ok(Self { model, tokenizer })
    }
    
    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        // Tokenize input
        let input_tokens = self.tokenizer.encode(prompt, true)?;
        
        // Run inference
        let output_tokens = self.model.generate(&input_tokens, max_tokens)?;
        
        // Detokenize output
        let output_text = self.tokenizer.decode(&output_tokens, true)?;
        
        Ok(output_text)
    }
}
```

### Benefits for libshimmy
1. ✅ **Standalone testing** - Can test tokenization separately
2. ✅ **Clear separation** - Tokenization vs inference
3. ✅ **Reusability** - Other projects can use shimmytok
4. ✅ **Faster iteration** - Develop in parallel

---

## Timeline Summary

```
Week 1 (Oct 20-26): GGUF vocab loading
  - Minimal GGUF reader
  - Vocabulary loading
  - Basic tests

Week 2 (Oct 27 - Nov 2): SentencePiece
  - Unicode normalization
  - Viterbi algorithm
  - Validation against llama.cpp

Week 3 (Nov 3-9): BPE
  - Merge extraction
  - BPE algorithm
  - GPT model testing

Week 4 (Nov 10-16): Edge cases
  - Multiple models
  - Unicode edge cases
  - Special token handling
  - Comprehensive test suite

Week 5 (Nov 17-23): Polish & release
  - Documentation
  - Examples
  - Benchmarks
  - CI/CD
  - Publish v0.1.0

Total: ~5 weeks to production-ready crate
```

---

## What Makes This Different from llama.cpp Port

### llama.cpp Approach
- 2000 LOC of tokenization code
- Mixed with model loading, inference
- C++ with lots of legacy baggage
- Hard to understand without deep knowledge

### shimmytok Approach
- ~1000 LOC clean Rust
- Standalone, focused on tokenization only
- Based on published algorithms, not llama.cpp internals
- Clear separation: GGUF loading vs algorithms vs API
- Documented against papers, not against llama.cpp

### Key Insight
**We're not porting llama.cpp's tokenization.**
**We're implementing SentencePiece and BPE that happen to work with GGUF files.**

The only llama.cpp-specific part is reading the GGUF vocab format (~200 LOC).

---

## Confidence Assessment

### What I KNOW (90%+ confidence)
- ✅ SentencePiece algorithm (published, well-understood)
- ✅ BPE algorithm (published, well-understood)
- ✅ Unicode normalization (standard, crate exists)
- ✅ Rust programming (we've built libshimmy successfully)
- ✅ Testing methodology (test against llama.cpp)

### What I Need to LEARN (60-75% confidence)
- ⚠️ GGUF vocab format details (2-3 days reverse-engineering)
- ⚠️ Special token edge cases (1 week testing)
- ⚠️ Model-specific quirks (ongoing discovery)
- ⚠️ BPE merge extraction (medium complexity)

### What I'll DISCOVER (unknown unknowns)
- 🔴 Undocumented GGUF variations
- 🔴 Edge cases in real-world models
- 🔴 Performance bottlenecks
- 🔴 Unicode corner cases

### Overall Confidence: 75%

**Why 75% is GOOD**:
- Quantization was 60% → delivered 100%
- SentencePiece/BPE are EASIER than quantization (algorithms are public)
- We can validate every step (test against llama.cpp)
- Scope is manageable (5 weeks, ~1000 LOC)

**Historical comparison**:
- libshimmy quantization: 60% confidence → ✅ Success
- libshimmy SIMD: 50% confidence → ✅ Success (better than port)
- shimmytok: 75% confidence → Likely ✅ Success

---

## Immediate Next Steps

### Step 1: Create Repository Structure (TODAY)
```bash
cd /c/Users/micha/repos/shimmytok
mkdir -p src tests examples benches fixtures
touch src/{lib.rs,vocab.rs,sentencepiece.rs,bpe.rs,normalize.rs,gguf_metadata.rs,error.rs}
touch tests/{test_vocab_loading.rs,test_sentencepiece.rs,test_bpe.rs}
touch examples/{basic_usage.rs,inspect_vocab.rs}
```

### Step 2: Setup Cargo.toml (TODAY)
```toml
[package]
name = "shimmytok"
version = "0.1.0"
edition = "2021"

[dependencies]
unicode-normalization = "0.1"
thiserror = "1.0"

[dev-dependencies]
criterion = "0.5"
tempfile = "3.0"
serde_json = "1.0"
```

### Step 3: Copy Test Model (TODAY)
```bash
cp ../libshimmy/models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf fixtures/
```

### Step 4: Stub Out Skeleton (TODAY - 30 min)
Create skeleton implementations with TODOs and type signatures

### Step 5: First Test (TODAY - 15 min)
Write failing test that shows intent:
```rust
#[test]
fn test_load_tinyllama() {
    let vocab = Vocabulary::from_gguf_file("fixtures/tinyllama...").unwrap();
    assert_eq!(vocab.len(), 32000);
}
```

### Step 6: Multi-Pass Audit (THIS WEEK)
Review this document 2-3 times, refine architecture before implementation

---

## Questions to Answer During Week 1

1. **GGUF Format**: What's the exact byte layout of metadata arrays?
2. **Special Tokens**: How does llama.cpp decide when to add BOS/EOS?
3. **Vocab Size**: Do all models follow same token_id → string mapping?
4. **Scores**: Are scores present for BPE models or only SentencePiece?
5. **Token Types**: What do each of the 6 token types mean in practice?

**How to Answer**: Read TinyLlama GGUF file, inspect with hex editor + existing tools

---

## Final Checklist Before Implementation

- [ ] This document reviewed 2+ times by user
- [ ] Architecture approved
- [ ] Repository structure created
- [ ] Skeleton code stubbed out
- [ ] Test fixtures prepared (TinyLlama GGUF copied)
- [ ] First failing tests written (showing intent)
- [ ] Dependencies added to Cargo.toml
- [ ] README drafted with goals

**THEN**: Begin Week 1 implementation (GGUF vocab loading)

---

## Success Metrics

### Week 1 Success
- [ ] Can read GGUF header and metadata
- [ ] Can load vocabulary arrays (tokens, scores, types)
- [ ] Can access tokens by ID and by string
- [ ] Tests pass for TinyLlama vocab loading

### Week 2 Success
- [ ] SentencePiece tokenizes simple text
- [ ] Output matches llama.cpp on basic examples
- [ ] Roundtrip works (encode → decode)
- [ ] Unicode normalization working

### Week 3 Success
- [ ] BPE algorithm implemented
- [ ] Works with at least one BPE model
- [ ] Merge extraction functional

### Week 4 Success
- [ ] Works with 3+ model types
- [ ] 50+ test cases passing
- [ ] Edge cases handled gracefully
- [ ] Matches llama.cpp on comprehensive test suite

### Week 5 Success
- [ ] Documentation complete
- [ ] Examples working and documented
- [ ] Benchmarks show adequate performance
- [ ] Published to crates.io as v0.1.0

---

## Long-Term Vision

### v0.1.0 (Week 5)
- SentencePiece + BPE
- Works with LLaMA, Phi, GPT models
- Basic API
- Published to crates.io

### v0.2.0 (Month 2)
- 10+ model architectures supported
- Performance optimizations (trie, SIMD)
- Batch tokenization API
- Community contributions

### v0.3.0 (Month 3)
- Custom tokenizer training (optional)
- Streaming tokenization
- More edge case handling
- Used by 10+ projects

### v1.0.0 (Month 6)
- Production-proven
- Used by 50+ projects
- Comprehensive model support
- THE Rust GGUF tokenization library

---

## Why This Document Exists

**Purpose**: Prevent scope creep, context loss, and architecture drift

**Rules**:
1. This is the **source of truth** for shimmytok
2. Review this document BEFORE every major decision
3. Update this document when architecture changes
4. Audit progress against this document weekly

**What This Prevents**:
- ❌ Starting implementation without clear plan
- ❌ Porting llama.cpp blindly
- ❌ Scope creep (adding unnecessary features)
- ❌ Context loss (forgetting why decisions were made)
- ❌ Architecture drift (straying from proven approach)

**What This Enables**:
- ✅ Clear, auditable progress
- ✅ Focused implementation
- ✅ Easy to onboard contributors
- ✅ Confidence in approach
- ✅ Fast iteration with validation

---

**TL;DR**: Create shimmytok as standalone GGUF tokenization crate. Implement published algorithms (SentencePiece, BPE), not port llama.cpp. 5 weeks to v0.1.0. First mover in Rust ecosystem. 75% confidence = GOOD odds.

**Current Status**: Repository created, master plan documented, ready for multi-pass audit before implementation.

**Next Action**: User audits this document 2-3 times, approves architecture, then we begin Week 1 (GGUF vocab loading).
