# shimmytok Roadmap

**Status**: v0.1.0 published to crates.io ✅  
**Primary Integration**: libshimmy (in development)  
**Maintenance Mode**: Stable API, future enhancements non-breaking

---

## 🎯 Project Mission

shimmytok is a **foundation library** - stable, focused, and free forever. The goal is to provide reliable pure Rust tokenization for GGUF models with 100% llama.cpp compatibility.

**Design Philosophy**:
- 🔒 API stability over feature velocity
- ✅ Correctness over performance (for now)
- 📦 Minimal dependencies over convenience
- 🧪 Test coverage over code coverage

---

## ✅ Completed (v0.1.0)

### Core Functionality
- [x] GGUF file parsing (v2, v3)
- [x] SentencePiece implementation with resegment algorithm
- [x] BPE implementation with priority queue merging
- [x] 100% llama.cpp validation (8/8 tests pass)
- [x] Unicode support (emoji, CJK, etc.)
- [x] Special token handling (BOS/EOS)

### API Surface (Stable - DO NOT BREAK)
- [x] `Tokenizer::from_gguf_file()` - Load from GGUF
- [x] `Tokenizer::encode()` - Text → Token IDs
- [x] `Tokenizer::decode()` - Token IDs → Text
- [x] `Tokenizer::vocab_size()` - Query vocabulary size
- [x] `Tokenizer::bos_token()` - Get BOS token ID
- [x] `Tokenizer::eos_token()` - Get EOS token ID
- [x] `TokenId = u32` type alias

### Supported Models
- [x] LLaMA / Llama-2 / Llama-3 (SentencePiece)
- [x] Mistral (SentencePiece) ✨ *v0.3.0*
- [x] Phi-3 (SentencePiece)
- [x] Qwen / Qwen2 (BPE) ✨ *v0.3.0*
- [x] Gemma (SentencePiece) ✨ *v0.3.0*
- [x] GPT-2 / GPT-3 (BPE)

### Infrastructure
- [x] Published to crates.io
- [x] GitHub Actions CI/CD
- [x] DCO enforcement
- [x] Comprehensive test suite (30 tests)
- [x] Documentation (README, CONTRIBUTING, etc.)
- [x] MIT license

---

## 🔮 Future Enhancements (Non-Breaking)

These additions **will not break** existing code using v0.1.x API.

### Phase 1: Performance Optimization (v0.2.0) ✅ **COMPLETED**
**Priority**: Medium  
**Impact**: 1.5-2x encode, 2-4x batch, ~40% overall speedup

- [x] **Parallel encoding** - `encode_batch()` method for multi-text encoding
  ```rust
  pub fn encode_batch(&self, texts: &[&str], add_special_tokens: bool) 
      -> Result<Vec<Vec<TokenId>>, Error>
  ```
  
- [x] **Vocabulary caching** - Already implemented (HashMap lookups)
  
- [ ] **SIMD optimization** - Fast string matching for common tokens *(deferred)*
  
- [x] **Benchmark suite** - Track performance across releases
  ```bash
  cargo bench --bench tokenization
  ```

**Actual Effort**: 8 Fibonacci points (estimated: 8)  
**Breaking Changes**: None  
**Testing**: All 31 tests passing, benchmarks established

---

### Phase 2: Model Support Expansion (v0.3.0) ✅ **COMPLETED**
**Priority**: Low  
**Impact**: Support 3 additional popular model families

- [x] **Mistral tokenizer** - Added "mistral" model type (SentencePiece)
  
- [x] **Qwen tokenizer** - Added "qwen"/"qwen2" model types (BPE)
  
- [x] **Gemma tokenizer** - Added "gemma" model type (SentencePiece)
  
- [x] **Model detection** - Query tokenizer type from metadata
  ```rust
  impl Tokenizer {
      pub fn model_type(&self) -> &str; // Returns "llama", "gpt2", etc.
  }
  ```

**Actual Effort**: 5 Fibonacci points (estimated: 13)  
**Breaking Changes**: None (additive only)  
**Testing**: All 31 tests passing, 7 model types supported

---

### Phase 3: Streaming & Advanced Features (v0.4.0)
**Priority**: Low  
**Impact**: Enable streaming use cases

- [ ] **Single token decode** - For streaming generation
  ```rust
  pub fn decode_single(&self, token: TokenId, skip_special_tokens: bool) 
      -> Result<String, Error>
  ```
  
- [ ] **Incremental encoder** - Stateful encoding for long texts
  ```rust
  pub struct IncrementalEncoder { /* ... */ }
  impl IncrementalEncoder {
      pub fn new(tokenizer: &Tokenizer) -> Self;
      pub fn push(&mut self, text: &str) -> Result<Vec<TokenId>, Error>;
      pub fn finalize(self) -> Vec<TokenId>;
  }
  ```
  
- [ ] **Token metadata** - Query token properties
  ```rust
  pub fn token_type(&self, token: TokenId) -> Option<TokenType>;
  pub fn is_control_token(&self, token: TokenId) -> bool;
  pub fn token_to_piece(&self, token: TokenId) -> Result<String, Error>;
  ```

**Estimated Effort**: 13 Fibonacci points  
**Breaking Changes**: None  
**Testing Required**: Streaming correctness tests

---

### Phase 4: Developer Experience (v0.5.0)
**Priority**: Low  
**Impact**: Better debugging and introspection

- [ ] **Detailed error context** - Include token position in errors
  
- [ ] **Debug mode** - Verbose tokenization logging
  ```rust
  pub struct DebugOptions {
      pub log_tokens: bool,
      pub log_merges: bool,
      pub log_special_tokens: bool,
  }
  pub fn from_gguf_file_debug<P>(path: P, opts: DebugOptions) -> Result<Self, Error>
  ```
  
- [ ] **Tokenizer inspector** - CLI tool for debugging
  ```bash
  cargo install shimmytok-cli
  shimmytok inspect model.gguf
  shimmytok encode "Hello world" model.gguf --verbose
  ```

**Estimated Effort**: 8 Fibonacci points  
**Breaking Changes**: None  
**Testing Required**: Integration tests for CLI

---

### Phase 5: WASM Support (v0.6.0)
**Priority**: Low  
**Impact**: Enable browser-based LLM applications

- [ ] **WASM compilation** - `wasm32-unknown-unknown` target
  
- [ ] **JavaScript bindings** - wasm-bindgen wrapper
  ```javascript
  import { Tokenizer } from 'shimmytok';
  const tokenizer = await Tokenizer.fromGGUF(modelBuffer);
  const tokens = tokenizer.encode("Hello world");
  ```
  
- [ ] **Browser example** - Demo tokenization in browser
  
- [ ] **npm package** - Publish to npm registry

**Estimated Effort**: 13 Fibonacci points  
**Breaking Changes**: None (additional target only)  
**Testing Required**: Browser compatibility tests

---

## 🚫 Non-Goals

These are **explicitly out of scope** for shimmytok:

### Will NOT Add
- ❌ **Training new tokenizers** - Use SentencePiece/Tiktoken directly
- ❌ **Converting between formats** - Use external tools
- ❌ **Model inference** - Use libshimmy or llama.cpp
- ❌ **Vocabulary modification** - Tokenizer is read-only
- ❌ **Custom tokenizer algorithms** - Stick to standard implementations
- ❌ **Python bindings** - Focus on Rust ecosystem first

### Why Not?
- **Scope creep**: shimmytok is a focused tokenization library
- **Maintenance burden**: Each feature adds testing/support overhead  
- **Better alternatives exist**: Training/inference have dedicated tools
- **Mission alignment**: Foundation libraries should do one thing well

---

## 🔒 API Stability Guarantee

### Contract with libshimmy

These APIs are **locked forever** (or until major version bump):

```rust
// NEVER BREAK THESE
pub struct Tokenizer;
pub type TokenId = u32;

impl Tokenizer {
    pub fn from_gguf_file<P: AsRef<Path>>(path: P) -> Result<Self, Error>;
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<TokenId>, Error>;
    pub fn decode(&self, tokens: &[TokenId], skip_special_tokens: bool) -> Result<String, Error>;
}

pub enum Error { /* ... */ }
impl std::fmt::Display for Error { /* ... */ }
impl std::error::Error for Error { /* ... */ }
```

### Semantic Versioning

- **Patch (0.1.x)**: Bug fixes, internal changes, documentation
- **Minor (0.x.0)**: New methods, new model types, performance improvements
- **Major (x.0.0)**: Breaking API changes (avoid if possible)

### Deprecation Policy

If a method must be deprecated:
1. Add `#[deprecated]` attribute in minor version
2. Maintain for at least 2 minor versions
3. Document migration path
4. Only remove in major version bump

---

## 📊 Success Metrics

### Quality Gates (Must Maintain)
- ✅ 100% test pass rate vs llama.cpp
- ✅ Zero clippy warnings
- ✅ Zero rustfmt diffs
- ✅ DCO sign-off on all commits
- ✅ Security audit pass (cargo audit)

### Performance Targets (Nice to Have)
- 📈 Within 2x of llama.cpp speed (C++)
- 📈 <10ms to load tokenizer from GGUF
- 📈 >10k tokens/sec encode throughput
- 📈 >50k tokens/sec decode throughput

### Adoption Metrics (Monitoring)
- 📦 Downloads from crates.io
- ⭐ GitHub stars
- 🐛 Issues opened vs resolved
- 💬 Community discussions
- 💝 Sponsorship growth

---

## 🤝 Contributing to Roadmap

### How to Propose Features

1. **Check existing issues/discussions** - Avoid duplicates
2. **Open GitHub Discussion** - Describe use case
3. **Wait for maintainer response** - May be out of scope
4. **Create detailed issue** - If approved
5. **Submit PR** - With tests + docs

### What Gets Accepted?

✅ **Good fit**:
- Maintains API stability
- Adds value for libshimmy integration
- Includes tests + docs
- Follows Rust best practices
- Non-breaking changes

❌ **Poor fit**:
- Breaks existing API
- Out of scope (training, inference, etc.)
- Adds heavy dependencies
- Platform-specific code (without good reason)
- Duplicates existing functionality

### Prioritization

Features are prioritized by:
1. **Correctness bugs** - Immediate fix
2. **libshimmy blockers** - High priority
3. **Security issues** - High priority
4. **Performance regressions** - Medium priority
5. **New features** - Low priority (unless sponsored)

---

## 🏗️ Development Status

### Current Focus (Q4 2025)
**Integration with libshimmy** - shimmytok is feature-complete for libshimmy v0.1.0

### Maintenance Mode (2026+)
**Stable API** - Only bug fixes and critical updates unless:
- Security vulnerability found
- New model type needed for libshimmy
- Community sponsorship for specific feature

### Long-Term Vision (3+ years)
**Reference implementation** - shimmytok as the canonical pure Rust tokenizer for GGUF models

---

## 💝 Sponsorship & Feature Requests

### Want a Feature Implemented?

**Option 1: Contribute it yourself**  
- Fork, implement, test, submit PR
- Maintainer will review for API compatibility

**Option 2: Sponsor development**  
- [GitHub Sponsors](https://github.com/sponsors/Michael-A-Kuykendall)
- $500/month tier gets roadmap input
- Larger sponsors can fund specific features

**Option 3: Wait**  
- Features may be implemented if/when maintainer has time
- No guarantees on timeline

### Current Sponsors
See [SPONSORS.md](SPONSORS.md) for our amazing sponsors! 💝

---

## 📅 Version History

| Version | Date | Highlights |
|---------|------|------------|
| v0.3.0 | Oct 2025 | Model expansion - Mistral, Qwen, Gemma support + `model_type()` method |
| v0.2.0 | Oct 2025 | Performance - `encode_batch()` parallel encoding, benchmarks, ~40% faster |
| v0.1.0 | Oct 2025 | Initial release - SentencePiece + BPE, published to crates.io |

---

## 🔗 Related Projects

- **[libshimmy](https://github.com/Michael-A-Kuykendall/libshimmy)** - Pure Rust LLM inference (uses shimmytok)
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** - Reference C++ implementation
- **[GGUF spec](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)** - File format documentation

---

## 📞 Contact

**Maintainer**: Michael A. Kuykendall  
**GitHub**: [@Michael-A-Kuykendall](https://github.com/Michael-A-Kuykendall)  
**Issues**: [shimmytok/issues](https://github.com/Michael-A-Kuykendall/shimmytok/issues)  
**Discussions**: [shimmytok/discussions](https://github.com/Michael-A-Kuykendall/shimmytok/discussions)

---

**Last Updated**: October 22, 2025  
**Next Review**: When libshimmy v0.1.0 ships  
**Maintained By**: [@Michael-A-Kuykendall](https://github.com/Michael-A-Kuykendall)
