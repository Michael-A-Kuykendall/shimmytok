# 🗺️ shimmytok Roadmap

**Current Version**: v0.7.0  
**Status**: Production Ready  
**Primary Use Case**: GGUF tokenization for libshimmy integration

---

## 🎯 Project Mission

shimmytok is a **pure Rust tokenizer library** for GGUF model files with 100% llama.cpp compatibility.

**Design Philosophy**:
- 🔒 **Correctness over performance** - Match llama.cpp exactly
- 📦 **Minimal dependencies** - thiserror + regex only
- 🦀 **Pure Rust** - No C++ bindings required
- ✅ **Validation-driven** - Every tokenizer validated against llama.cpp

---

## ✅ v0.7.0 - Full llama.cpp Parity (Current)

### Tokenizer Coverage
| Type | Name | Status | Validated Against |
|------|------|--------|-------------------|
| SPM | SentencePiece | ✅ | llama-spm |
| BPE | Byte-Pair Encoding | ✅ | gpt-2, starcoder, qwen2, deepseek-coder, deepseek-llm, falcon, command-r, refact |
| WPM | Word-Piece Model | ✅ | bert-bge |
| UGM | Unigram | ✅ | *(implementation complete, needs T5 GGUF)* |
| RWKV | RWKV World | ✅ | *(implementation complete, needs model)* |
| - | PLaMo-2 | ✅ | *(no GGUF available from llama.cpp)* |

### Validated Models (10/10 passing)
All models validated against `llama-tokenize` binary with exact token match:

- ✅ `ggml-vocab-bert-bge.gguf` (WPM)
- ✅ `ggml-vocab-command-r.gguf` (BPE)
- ✅ `ggml-vocab-deepseek-coder.gguf` (BPE)
- ✅ `ggml-vocab-deepseek-llm.gguf` (BPE)
- ✅ `ggml-vocab-falcon.gguf` (BPE)
- ✅ `ggml-vocab-gpt-2.gguf` (BPE)
- ✅ `ggml-vocab-llama-spm.gguf` (SPM)
- ✅ `ggml-vocab-qwen2.gguf` (BPE)
- ✅ `ggml-vocab-refact.gguf` (BPE)
- ✅ `ggml-vocab-starcoder.gguf` (BPE)

### Public API (Stable)
```rust
// Core API
Tokenizer::from_gguf_file(path) -> Result<Tokenizer>
tokenizer.encode(text, add_special_tokens) -> Result<Vec<TokenId>>
tokenizer.decode(&token_ids) -> Result<String>
tokenizer.decode_single(token_id) -> Result<String>

// Metadata
tokenizer.vocab_size() -> usize
tokenizer.bos_token() -> Option<TokenId>
tokenizer.eos_token() -> Option<TokenId>
tokenizer.model_type() -> &str
tokenizer.pre_type() -> &str

// Batch processing
tokenizer.encode_batch(texts, add_special) -> Result<Vec<Vec<TokenId>>>
```

---

## 🔮 v0.8.0 - Extended Validation

**Target**: Additional tokenizer validation with available models

### Planned
- [ ] **RWKV validation** - Test with `rwkv-4-pile-169m` GGUF when available
- [ ] **T5/UGM validation** - Investigate llama.cpp T5 architecture support
- [ ] **Additional BPE patterns** - Any new vocab files from llama.cpp updates

### Stretch Goals
- [ ] **Phi-4 validation** - When GGUF available
- [ ] **Llama-3.1/3.2 validation** - Verify continued compatibility

---

## 🌟 Future Considerations

### May Implement
- **SIMD optimization** - Performance without sacrificing correctness
- **Streaming encoder** - Token-by-token encoding for very large texts
- **Async file loading** - Non-blocking GGUF parsing

### Will Not Implement
- ❌ C++ dependencies or FFI
- ❌ Training/fine-tuning support
- ❌ Model inference capabilities
- ❌ Tokenizer training from scratch
- ❌ Non-GGUF format support (safetensors, etc.)

---

## 📊 Version History

| Version | Date | Highlights |
|---------|------|------------|
| v0.7.0 | Jan 2025 | Full llama.cpp parity: WPM, UGM, RWKV, PLaMo-2 |
| v0.6.0 | Jan 2025 | llama.cpp validation fixes |
| v0.4.0 | Oct 2024 | Streaming decode, token introspection |
| v0.3.0 | Oct 2024 | Mistral, Qwen, Gemma support |
| v0.2.0 | Oct 2024 | Batch encoding, benchmarks |
| v0.1.0 | Oct 2024 | Initial release: SPM + BPE |

---

## 🔗 Related Projects

- **[libshimmy](https://github.com/Michael-A-Kuykendall/libshimmy)** - Pure Rust LLM inference (uses shimmytok)
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** - Reference C++ implementation
- **[GGUF spec](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)** - File format documentation

---

**Maintainer**: Michael A. Kuykendall  
**License**: MIT  
**Last Updated**: January 2025
