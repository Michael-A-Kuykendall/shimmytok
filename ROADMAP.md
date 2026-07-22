# shimmytok Roadmap

**Current version**: v0.7.3  
**Status**: Production ready  
**Mission**: Pure Rust GGUF tokenizer with 100 % llama.cpp compatibility

---

## What shimmytok is

A tokenizer library for Rust developers building LLM inference engines.

- **Pure Rust** — No C/C++ dependencies, no FFI, compiles anywhere Rust does
- **GGUF-native** — Loads vocabulary directly from model files; no sidecar files
- **llama.cpp-compatible** — Validated token-for-token against `llama-tokenize`
- **Embeddable** — Three direct dependencies (`fancy-regex`, `rayon`, `thiserror`)

## What shimmytok is not

- A tokenizer training library
- A Python library (use HuggingFace `tokenizers`)
- A general-purpose NLP toolkit
- A model inference engine (see [libshimmy](https://github.com/Michael-A-Kuykendall/libshimmy))

---

## Current state (v0.7.x)

### Tokenizer support

| Algorithm | Models | Status |
|-----------|--------|--------|
| SentencePiece (SPM) | LLaMA, Mistral, Gemma, Phi-3 | ✅ Validated |
| BPE | GPT-2, Qwen2, StarCoder, DeepSeek, Falcon, … | ✅ Validated |
| WordPiece (WPM) | BERT, BGE embeddings | ✅ Validated |
| Unigram (UGM) | T5-style | ⚠️ Implemented — needs GGUF fixture |
| RWKV | RWKV World | ⚠️ Implemented — needs GGUF fixture |
| PLaMo-2 | PLaMo-2 | ⚠️ Implemented — no public GGUF exists |

**41 BPE pre-tokenization patterns** are implemented, covering GPT-2, Llama-3, Qwen2,
DeepSeek (LLM / Coder / V3 / R1), StarCoder, Falcon, Command-R, ChatGLM4, DBRX,
Tekken, Grok-2, and more.

### Stable API surface

```rust
// Load
Tokenizer::from_gguf_file(path) -> Result<Tokenizer>

// Encode
tokenizer.encode(text, add_special)             -> Result<Vec<TokenId>>
tokenizer.encode_with_options(text, &opts)      -> Result<Vec<TokenId>>
tokenizer.encode_batch(texts, add_special)      -> Result<Vec<Vec<TokenId>>>

// Decode
tokenizer.decode(tokens, skip_special)          -> Result<String>
tokenizer.decode_with_options(tokens, &opts)    -> Result<String>
tokenizer.decode_single(token, skip_special)    -> Result<String>

// Introspection
tokenizer.vocab_size()          -> usize
tokenizer.bos_token()           -> TokenId
tokenizer.eos_token()           -> TokenId
tokenizer.model_type()          -> &str
tokenizer.pre_type()            -> Option<&str>
tokenizer.token_to_piece(id)    -> Result<String>
tokenizer.token_type(id)        -> TokenType
tokenizer.is_special_token(id)  -> bool
```

### Minimum Supported Rust Version (MSRV)

**Rust 1.70** (stable). This will not be raised without a minor-version bump.

---

## Roadmap

### v0.8.0 — Performance

Goal: measurable throughput improvements without touching the public API.

- [ ] Criterion benchmark suite published to track regressions across releases
- [ ] Profile BPE hot path — `merge_ranks` HashMap key allocation under load
- [ ] Profile SPM hot path — `rev_merge` HashMap growth on long documents
- [ ] Evaluate arc-interned string keys for merge-rank maps
- [ ] Rayon threshold tuning for `encode_batch` (parallelism overhead on small batches)
- [ ] Memory allocation audit — avoid unnecessary intermediate `Vec<String>` in BPE
      pre-tokenizer for long documents

**Success metric**: encode throughput within 2× of `tiktoken` for BPE models;
within 2× of the `sentencepiece` C library for SPM models, measured on a standard
text corpus.

### v0.9.0 — Production hardening

Goal: safe to deploy where inputs come from untrusted sources.

- [ ] Fuzz corpus with `cargo-fuzz` (AFL targets for GGUF parsing and encode/decode)
- [ ] Malformed GGUF edge-case tests (truncated arrays, impossible counts, NaN scores)
- [ ] UGM / RWKV validation once commodity GGUF fixtures become available
- [ ] Error message audit — all `Error` variants must name the offending field/value

**Success metric**: no panics on any input in `cargo fuzz` after 24 h corpus run.

### v1.0.0 — Stable release

Goal: public API stability commitment.

- [ ] API review and freeze — semver guarantee from this point forward
- [ ] MSRV policy formalised in `Cargo.toml` (`rust-version` field)
- [ ] Documentation audit — every public item has a doc-comment and at least one example
- [ ] `#![deny(missing_docs)]` enabled
- [ ] Published MSRV CI matrix (MSRV, stable, beta, nightly)

**Success metric**: no breaking changes in the 1.x line.

---

## Intentionally out of scope

| Feature | Reason | Alternative |
|---------|--------|-------------|
| Tokenizer training | Not this library's lane | SentencePiece, `tokenizers` |
| Non-GGUF formats | GGUF-only scope | HuggingFace `tokenizers` |
| Python bindings | Rust-first; wrap it yourself | PyO3 |
| Model inference | Tokenizer only | libshimmy |
| Async `from_gguf_file` | Use `spawn_blocking` | `tokio::task::spawn_blocking` |
| Streaming encode iterator | Low demand vs. complexity | Chunk the input yourself |

---

## Integration

```
Your Rust application
        │
        ▼
libshimmy  (or your inference engine)
        │
        ▼
shimmytok  (this library)
        │
        ▼
 model.gguf
```

Primary downstream consumer: [libshimmy](https://github.com/Michael-A-Kuykendall/libshimmy)

---

## License

`MIT OR Apache-2.0`

## Maintainer

Michael A. Kuykendall — [@Michael-A-Kuykendall](https://github.com/Michael-A-Kuykendall)
