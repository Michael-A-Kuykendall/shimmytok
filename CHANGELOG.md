# Changelog

All notable changes to shimmytok will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-10-22

### Added
- `encode_batch()` method for parallel encoding of multiple texts
- Comprehensive benchmark suite using Criterion
- Thread safety via `Send + Sync` bounds on `TokenizerImpl`

### Performance
- 1.5-2x speedup on `encode()` (vocabulary caching already in place)
- 2-4x speedup on batch encoding via Rayon parallel processing
- ~40% improvement on decode and load operations

### Internal
- Added rayon dependency for parallel batch processing
- Added criterion and dirs dev dependencies for benchmarking
- Benchmarks for encode, decode, load, and batch operations

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
  - `decode_single()` - Single token to text
  - `vocab_size()` - Get vocabulary size
  - `token_to_piece()` - Get token bytes
- Comprehensive error handling with `thiserror`
- Input validation (MAX_INPUT_SIZE: 10MB, MAX_OUTPUT_TOKENS: 1M)
- 30 tests including:
  - 8/8 llama.cpp validation tests (100% match)
  - 7 negative/error handling tests
  - Round-trip verification
  - Unicode/emoji handling
  - Special token (BOS/EOS) handling
- No unsafe code - pure safe Rust
- Minimal dependencies (thiserror + regex only)

### Validated Models
- ✅ LLaMA/Llama-2/Llama-3 (SentencePiece)
- ✅ Phi-3 (SentencePiece)
- ✅ GPT-2 (BPE)

### Documentation
- Comprehensive README with examples
- API documentation with rustdoc
- Contributing guidelines
- Code of Conduct
- Security policy
- DCO (Developer Certificate of Origin)

[Unreleased]: https://github.com/Michael-A-Kuykendall/shimmytok/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Michael-A-Kuykendall/shimmytok/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Michael-A-Kuykendall/shimmytok/releases/tag/v0.1.0
