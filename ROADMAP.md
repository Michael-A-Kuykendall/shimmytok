# shimmytok Roadmap

**Current Version**: v0.7.1  
**Status**: Production Ready  
**Mission**: Pure Rust GGUF tokenizer with 100% llama.cpp compatibility

---

## What shimmytok Is

A tokenizer library for Rust developers building LLM inference engines.

- **Pure Rust** - No C/C++ dependencies, no FFI
- **GGUF Native** - Loads vocab directly from model files
- **llama.cpp Compatible** - Validated against `llama-tokenize`
- **Embeddable** - Single dependency for Rust inference projects

## What shimmytok Is NOT

- A tokenizer training library
- A Python library (use HuggingFace tokenizers)
- A general-purpose NLP toolkit
- A model inference engine (that's libshimmy)

---

## Current State (v0.7.x)

### Tokenizer Support

| Algorithm | Models | Status |
|-----------|--------|--------|
| SentencePiece | Llama, Mistral, Gemma | Validated |
| BPE | GPT-2, Qwen, StarCoder, DeepSeek | Validated |
| WPM | BERT | Validated |
| UGM | T5-style | Implemented, needs GGUF model |
| RWKV | RWKV World | Implemented, needs GGUF model |
| PLaMo-2 | PLaMo | Implemented, no GGUF exists |

### API Surface (Stable)

```rust
// Load
Tokenizer::from_gguf_file(path) -> Result<Tokenizer>

// Encode
tokenizer.encode(text, add_special) -> Result<Vec<u32>>
tokenizer.encode_batch(texts, add_special) -> Result<Vec<Vec<u32>>>

// Decode  
tokenizer.decode(tokens, skip_special) -> Result<String>
tokenizer.decode_single(token) -> Result<String>

// Metadata
tokenizer.vocab_size() -> usize
tokenizer.bos_token() -> u32
tokenizer.eos_token() -> u32
tokenizer.model_type() -> &str
```

---

## Roadmap

### v0.8.0 - Performance

Focus: Make it fast without breaking correctness.

- [ ] Benchmark suite against HuggingFace tokenizers
- [ ] Profile hot paths (vocab lookup, BPE merge)
- [ ] Consider SIMD for batch operations
- [ ] Memory allocation optimization

Success metric: Within 2x of tiktoken for BPE, within 2x of sentencepiece for SPM.

### v0.9.0 - Production Hardening

Focus: Ready for untrusted input in production.

- [ ] Fuzz testing with cargo-fuzz
- [ ] Memory limit enforcement (already started)
- [ ] Malformed GGUF handling
- [ ] Error message improvements

Success metric: No panics on any input, clear error messages.

### v1.0.0 - Stable Release

Focus: API stability commitment.

- [ ] API review and freeze
- [ ] Documentation audit
- [ ] crates.io publish with stable guarantees
- [ ] MSRV policy (minimum supported Rust version)

Success metric: Semver commitment, no breaking changes in 1.x.

---

## Out of Scope (Will Not Implement)

These are intentionally excluded to keep shimmytok focused:

| Feature | Reason | Alternative |
|---------|--------|-------------|
| Tokenizer training | Not our lane | SentencePiece, tokenizers |
| Non-GGUF formats | GGUF-only scope | HuggingFace tokenizers |
| Python bindings | Rust-first library | PyO3 wrapper by others |
| Model inference | Tokenizer only | libshimmy |
| Streaming encode | Complexity vs value | Chunk input yourself |
| Async loading | Complexity vs value | spawn_blocking |

---

## Integration

shimmytok is designed for:

```
Your Rust App
     |
     v
libshimmy (inference)  <-- or your own inference code
     |
     v
shimmytok (tokenization)
     |
     v
model.gguf (GGUF file)
```

Primary consumer: [libshimmy](https://github.com/Michael-A-Kuykendall/libshimmy)

---

## Version History

| Version | Date | Focus |
|---------|------|-------|
| 0.7.1 | Jan 2026 | Code quality, test coverage |
| 0.7.0 | Jan 2025 | Full llama.cpp parity |
| 0.6.0 | Jan 2025 | Validation fixes |
| 0.4.0 | Oct 2024 | Streaming decode |
| 0.3.0 | Oct 2024 | Multi-model support |
| 0.2.0 | Oct 2024 | Batch encoding |
| 0.1.0 | Oct 2024 | Initial release |

---

## License

MIT OR Apache-2.0

## Maintainer

Michael A. Kuykendall
