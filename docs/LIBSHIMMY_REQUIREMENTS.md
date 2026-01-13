# libshimmy Requirements for shimmytok

**Date**: 2025-10-21  
**Purpose**: Lock down the minimum API surface that libshimmy depends on  
**Status**: Contract - DO NOT BREAK these APIs

---

## Critical: Must Not Change

These are the exact APIs libshimmy integration depends on. Breaking changes here break libshimmy.

### 1. Tokenizer Struct

```rust
pub struct Tokenizer {
    // Internal fields can change, just maintain the API
}
```

**Required Methods**:

```rust
impl Tokenizer {
    /// Load tokenizer from GGUF file
    /// libshimmy calls this in Model::load()
    pub fn from_gguf_file<P: AsRef<Path>>(path: P) -> Result<Self, Error>
    
    /// Encode text to token IDs
    /// libshimmy calls this in Generator::generate() for user prompts
    /// add_special_tokens: libshimmy will pass `true` for initial prompt, `false` for continuations
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<TokenId>, Error>
    
    /// Decode token IDs back to text
    /// libshimmy calls this to convert generated tokens back to strings
    /// skip_special_tokens: libshimmy will pass `true` to strip BOS/EOS from output
    pub fn decode(&self, tokens: &[TokenId], skip_special_tokens: bool) -> Result<String, Error>
}
```

### 2. Type Aliases

```rust
/// libshimmy uses u32 for token IDs
pub type TokenId = u32;
```

### 3. Error Type

```rust
/// libshimmy will convert these to anyhow::Error
pub enum Error {
    // Variants can change, libshimmy just needs Display impl
}

impl std::fmt::Display for Error { ... }
impl std::error::Error for Error { ... }
```

---

## Validation Requirements

### Correctness

**Must match llama.cpp output exactly for**:
- LLaMA/Llama-2/Llama-3 models (SentencePiece)
- All test cases in `tests/test_comprehensive.rs`
- Unicode handling (emoji, Chinese, etc.)
- Special tokens (BOS/EOS placement)

**Test Command**:
```bash
cargo test test_against_llama_cpp -- --nocapture
```

**Success Criteria**: 8/8 tests pass with "✓ MATCH"

### Model Support

**Required**:
- ✅ "llama" tokenizer type (SentencePiece)

**Optional** (nice to have, not blocking):
- ⚠️ "gpt2" tokenizer type (BPE)
- Other model types

---

## Nice to Have (Not Required)

These would be useful but libshimmy can work around them if missing.

### 1. Special Token Queries

```rust
impl Tokenizer {
    /// Query BOS token ID
    /// libshimmy can parse from GGUF metadata directly if this doesn't exist
    pub fn bos_token(&self) -> Option<TokenId> { ... }
    
    /// Query EOS token ID
    /// libshimmy can parse from GGUF metadata directly if this doesn't exist
    pub fn eos_token(&self) -> Option<TokenId> { ... }
    
    /// Query vocabulary size
    /// libshimmy can get from GGUF metadata if this doesn't exist
    pub fn vocab_size(&self) -> usize { ... }
}
```

**Impact if missing**: Minor - libshimmy will duplicate GGUF parsing for these values

### 2. Batch Encoding

```rust
impl Tokenizer {
    /// Encode multiple texts at once
    /// libshimmy doesn't need this yet but would use it for batch processing
    pub fn encode_batch(&self, texts: &[&str], add_special_tokens: bool) -> Result<Vec<Vec<TokenId>>, Error>
}
```

**Impact if missing**: None currently - libshimmy calls `encode()` in loop

### 3. Token-by-Token Decoding

```rust
impl Tokenizer {
    /// Decode single token (for streaming)
    /// libshimmy could use this for streaming generation
    pub fn decode_token(&self, token: TokenId) -> Result<String, Error>
}
```

**Impact if missing**: None currently - libshimmy buffers full response

---

## Integration Example

This is how libshimmy will use shimmytok:

```rust
// In libshimmy/src/tokenizer/mod.rs
use shimmytok;

pub struct Tokenizer {
    inner: shimmytok::Tokenizer,
}

impl Tokenizer {
    pub fn from_gguf(path: &Path) -> Result<Self> {
        let inner = shimmytok::Tokenizer::from_gguf_file(path)
            .map_err(|e| anyhow::anyhow!("Tokenizer load failed: {}", e))?;
        Ok(Self { inner })
    }
    
    pub fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        self.inner.encode(text, add_bos)
            .map_err(|e| anyhow::anyhow!("Encode failed: {}", e))
    }
    
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.inner.decode(tokens, true) // skip_special_tokens = true
            .map_err(|e| anyhow::anyhow!("Decode failed: {}", e))
    }
}

// In libshimmy/src/api/mod.rs
impl Model {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        // ... load GGUF metadata, weights ...
        let tokenizer = Tokenizer::from_gguf(path.as_ref())?;
        // ...
    }
}

impl Generator {
    pub fn generate(&mut self, prompt: &str, params: GenerationParams) -> Result<String> {
        // Tokenize prompt
        let mut tokens = self.tokenizer.encode(prompt, true)?; // add BOS
        
        // Generate loop
        for _ in 0..params.max_tokens {
            let logits = self.forward(&tokens)?;
            let next_token = sample(&logits, &params);
            
            if next_token == self.eos_token {
                break;
            }
            
            tokens.push(next_token);
        }
        
        // Detokenize (strips BOS/EOS)
        self.tokenizer.decode(&tokens)
    }
}
```

---

## Dependency Specification

**Cargo.toml** (libshimmy will use):
```toml
[dependencies]
shimmytok = { version = "0.1", path = "../shimmytok" }
# Later when published:
# shimmytok = "0.1"
```

---

## Testing Contract

libshimmy will NOT test tokenization correctness (shimmytok's job).

libshimmy WILL test:
- ✅ Can load tokenizer from GGUF
- ✅ Can encode/decode roundtrip
- ✅ Integration with inference engine
- ✅ End-to-end text generation

shimmytok MUST test:
- ✅ Token IDs match llama.cpp exactly
- ✅ Unicode handling
- ✅ Special token handling
- ✅ Edge cases (empty string, long text, etc.)

---

## Version Compatibility

**Breaking Changes** (require libshimmy update):
- Changing `Tokenizer::from_gguf_file()` signature
- Changing `encode()` or `decode()` signatures
- Changing `TokenId` type
- Removing any of the three required methods

**Non-Breaking Changes** (safe):
- Adding new methods
- Adding new error variants
- Internal implementation changes
- Performance improvements
- Bug fixes

**Versioning**: Follow semver
- Patch (0.1.x): Bug fixes, internal changes
- Minor (0.x.0): New features, non-breaking additions
- Major (x.0.0): Breaking API changes

---

## Current Status

**shimmytok v0.1.0**:
- ✅ All 3 required methods implemented
- ✅ 8/8 tests pass against llama.cpp
- ✅ Compiles with 3 warnings (unused metadata fields)
- ⚠️ Missing special token query methods (libshimmy can work around)
- ⚠️ BPE stub only (not needed for libshimmy's LLaMA focus)

**Ready for integration**: YES

---

## Summary

**Must maintain forever**:
1. `Tokenizer::from_gguf_file()`
2. `Tokenizer::encode()`
3. `Tokenizer::decode()`
4. `TokenId = u32`
5. Test pass rate: 100% vs llama.cpp

**Can change freely**:
- Internal implementation
- Additional methods
- Performance optimizations
- Documentation
- Examples
- Error messages (as long as Error trait implemented)

---

**END OF REQUIREMENTS**
