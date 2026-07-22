# shimmytok — Current Status

## ✅ v0.7.1 — Released and complete

All planned work has shipped. The codebase is in maintenance mode.

### What's done
- All six tokenizer algorithms: SPM, BPE, WPM, UGM, RWKV, PLaMo-2
- 41 BPE pre-tokenization patterns (full llama.cpp coverage)
- 10/10 validated models produce exact token match against `llama-tokenize`
- Dual MIT/Apache-2.0 license
- Invariant assertions in debug builds
- Property-based tests (proptest)
- Zero clippy warnings
- `#[non_exhaustive]` Error enum
- `regex` crate dependency removed (only `fancy-regex` used)

### Known limitations (non-blocking)
- UGM, RWKV, PLaMo-2 lack commodity GGUF test fixtures — marked `⚠️ Experimental` in docs
- UGM normalization is a placeholder (full llama.cpp charsmap/XCDA not yet ported)

### Next potential work
- Streaming encode API (zero-copy iterator over token IDs)
- WASM target validation
- Expand BPE pattern coverage as new models appear in llama.cpp
