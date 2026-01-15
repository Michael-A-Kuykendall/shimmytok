# Changelog

All notable changes to shimmytok will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.1] - 2026-01-15

### 🛡️ Code Quality & Testing

This release focuses on code quality, documentation, and test coverage improvements.

### Added

- **Invariants module** (`src/invariants.rs`) - Debug-mode runtime assertions for tokenizer correctness
- **Property-based testing** (`tests/test_proptest.rs`) - 6 proptest properties for fuzz testing
- **Coverage tests** (`tests/test_coverage.rs`) - 37 targeted tests for edge cases
- **Experimental warnings** - Clear documentation that PLaMo-2, RWKV, UGM modules lack commodity test fixtures

### Changed

- **Dual license** - Now `MIT OR Apache-2.0` with separate license files
- **Enhanced documentation** - Improved module-level rustdoc for `byte_encoder.rs` and `gguf.rs`
- **CODEOWNERS** moved to `.github/CODEOWNERS` (standard location)

### Fixed

- All clippy warnings resolved (10+ fixes)
- `const` assertion in plamo2.rs
- Invariant assertions no longer panic on intentionally-invalid user input in tests

### Testing

- Test coverage improved from ~62% to ~80%+ on core modules
- 22 unit tests + 37 integration tests + 6 proptest properties
- All tests pass with `cargo test`

## [0.7.0] - 2025-01-26

### 🎯 Full llama.cpp Tokenizer Parity

This release completes support for all tokenizer types in llama.cpp's `LLAMA_VOCAB_TYPE` enum.

### Added

**New Tokenizers**
- **WPM (Word-Piece Model)** - BERT-style tokenizer with phantom space and greedy longest match
- **RWKV** - Trie-based greedy matching with escape sequence support  
- **UGM (Unigram)** - Viterbi-style dynamic programming for optimal tokenization
- **PLaMo-2** - Table-driven reverse DP with byte fallback

**API Additions**
- `pre_type()` method - Query pre-tokenization pattern type
- `clean_spaces` decoding - llama.cpp parity for punctuation/contraction spacing
- `InvalidUtf8` error variant - Better error handling for decode operations

### Validated

All tokenizers validated against `llama-tokenize` with exact token match:

| Model | Type | Status |
|-------|------|--------|
| bert-bge | WPM | ✅ |
| command-r | BPE | ✅ |
| deepseek-coder | BPE | ✅ |
| deepseek-llm | BPE | ✅ |
| falcon | BPE | ✅ |
| gpt-2 | BPE | ✅ |
| llama-spm | SPM | ✅ |
| qwen2 | BPE | ✅ |
| refact | BPE | ✅ |
| starcoder | BPE | ✅ |

### Fixed
- `deepseek-llm` regex pattern simplified for Rust regex compatibility
- UGM `user_defined_trie` now correctly preprocesses text before Viterbi DP

## [0.6.0] - 2025-01-20

### Fixed
- llama.cpp parity fixes for Tier 1 models (13 tests passing)

## [0.4.0] - 2025-10-22

### Added
- `decode_single()` method for streaming token-by-token decoding
- `token_to_piece()` method to get raw token text
- `token_type()` method to query token classification
- `is_special_token()` method to check if token is special
- Comprehensive streaming test suite (6 new tests)

### Use Cases
- Real-time streaming generation support
- Token-level debugging and introspection
- Efficient single-token decoding for LLM streaming

## [0.3.0] - 2025-10-22

### Added
- Support for Mistral models (SentencePiece tokenizer)
- Support for Qwen/Qwen2 models (BPE tokenizer)
- Support for Gemma models (SentencePiece tokenizer)
- `model_type()` method to query tokenizer model type

## [0.2.0] - 2025-10-22

### Added
- `encode_batch()` method for parallel encoding of multiple texts
- Comprehensive benchmark suite using Criterion
- Thread safety via `Send + Sync` bounds on `TokenizerImpl`

### Performance
- 1.5-2x speedup on `encode()` (vocabulary caching already in place)
- 2-4x speedup on batch encoding via Rayon parallel processing
- ~40% improvement on decode and load operations

## [0.1.0] - 2025-10-22

### Added
- Initial release of shimmytok
- **SentencePiece** tokenization with `resegment()` algorithm (100% llama.cpp compatible)
- **BPE (Byte-Pair Encoding)** with priority queue merging and regex pre-tokenization
- GGUF v2 and v3 format support
- Load tokenizers directly from GGUF model files
- Public API:
  - `Tokenizer::from_gguf_file()` - Load from GGUF
  - `encode()` - Text to token IDs
  - `decode()` - Token IDs to text
  - `vocab_size()` - Get vocabulary size
  - `bos_token()` / `eos_token()` - Special tokens
- Comprehensive error handling with `thiserror`
- 30 tests with 100% llama.cpp match

### Validated Models
- ✅ LLaMA/Llama-2/Llama-3 (SentencePiece)
- ✅ Phi-3 (SentencePiece)
- ✅ GPT-2 (BPE)

[Unreleased]: https://github.com/Michael-A-Kuykendall/shimmytok/compare/v0.7.1...HEAD
[0.7.1]: https://github.com/Michael-A-Kuykendall/shimmytok/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/Michael-A-Kuykendall/shimmytok/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/Michael-A-Kuykendall/shimmytok/compare/v0.4.0...v0.6.0
[0.4.0]: https://github.com/Michael-A-Kuykendall/shimmytok/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Michael-A-Kuykendall/shimmytok/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Michael-A-Kuykendall/shimmytok/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Michael-A-Kuykendall/shimmytok/releases/tag/v0.1.0
