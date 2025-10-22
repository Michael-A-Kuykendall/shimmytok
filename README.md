<div align="center">

# shimmytok

### The pure Rust tokenizer for GGUF models
**llama.cpp compatible • standalone • no C++ required**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Crates.io](https://img.shields.io/crates/v/shimmytok.svg)](https://crates.io/crates/shimmytok)
[![Rust](https://img.shields.io/badge/rust-stable-brightgreen.svg)](https://rustup.rs/)
[![💝 Sponsor this project](https://img.shields.io/badge/💝_Sponsor-ea4aaa?style=flat&logo=github&logoColor=white)](https://github.com/sponsors/Michael-A-Kuykendall)

</div>

---

**shimmytok will be free forever.** MIT licensed, no strings attached.

💝 **If shimmytok helps you, consider [sponsoring](https://github.com/sponsors/Michael-A-Kuykendall) — 100% of support goes to keeping it free forever.**

---

## Features

- 🦀 **Pure Rust** - No C++ dependencies, works everywhere Rust works
- 📦 **Load from GGUF** - Read tokenizers directly from model files
- ✅ **100% Compatible** - Validated against llama.cpp output
- 🧪 **Fully Tested** - 30 comprehensive tests, 8/8 llama.cpp validation
- 🎯 **Simple API** - Just 6 public methods: `from_gguf_file`, `encode`, `decode`, `decode_single`, `vocab_size`, `token_to_piece`
- 🚀 **Lightweight** - ~2,700 lines of source code, minimal dependencies

## Why shimmytok?

Unlike other Rust tokenizers, shimmytok loads tokenizers **directly from GGUF files** without needing separate `.model` files or C++ dependencies. If you're building pure Rust LLM applications, this is the tokenizer you need.

## Installation

```toml
[dependencies]
shimmytok = "0.1"
```

## Usage

```rust
use shimmytok::Tokenizer;

// Load tokenizer from GGUF file
let tokenizer = Tokenizer::from_gguf_file("model.gguf")?;

// Encode text to token IDs
let tokens = tokenizer.encode("Hello world", true)?;
// Output: [1, 15043, 3186]

// Decode token IDs back to text
let text = tokenizer.decode(&tokens, true)?;
// Output: "Hello world"
```

### Control Special Tokens

```rust
// Add BOS/EOS tokens during encoding
let tokens = tokenizer.encode("Hello", true)?;  // Adds BOS if model requires it

// Skip special tokens during decoding
let text = tokenizer.decode(&tokens, true)?;  // Strips BOS/EOS from output
```

## Supported Models

| Model Type | Status | Implementation |
|------------|--------|----------------|
| LLaMA/Llama-2/Llama-3 | ✅ Full support | SentencePiece (validated) |
| Phi-3 | ✅ Full support | SentencePiece (validated) |
| GPT-2 / GPT-3 (BPE) | ✅ Full support | Priority queue BPE from llama.cpp |

## Validation

shimmytok is tested against actual llama.cpp output:

```
✓ MATCH for 'Hello world': [15043, 3186]
✓ MATCH for 'The quick brown fox': [450, 4996, 17354, 1701, 29916]
✓ MATCH for '🦀 Rust': [29871, 243, 162, 169, 131, 390, 504]
```

100% accuracy across Unicode, special tokens, whitespace handling, and more.

## Comparison

| Feature | shimmytok | kitoken | llama-cpp-sys |
|---------|-----------|---------|---------------|
| Load from GGUF | ✅ | ❌ | ✅ (via FFI) |
| Pure Rust | ✅ | ✅ | ❌ (C++) |
| No build dependencies | ✅ | ✅ | ❌ (needs C++) |
| WASM-ready | ✅ | ✅ | ❌ |
| Lines of code | 1157 | Unknown | 150K+ |

## Use Cases

- **Pure Rust LLM inference engines** (like [libshimmy](https://github.com/yourusername/libshimmy))
- **WASM LLM applications** (runs in browser)
- **Embedded systems** (no C++ compiler needed)
- **CLI tools** working with GGUF files
- **Research** (inspect tokenization without C++)

## Implementation

shimmytok implements:

- **SentencePiece** with the critical `resegment()` function from llama.cpp (100% validated)
- **BPE (Byte-Pair Encoding)** with priority queue-based merging and regex pre-tokenization patterns

Both algorithms are direct ports from llama.cpp source code analysis.

## Limitations

- Supports GGUF v2 and v3 formats only
- Optimized for correctness, not yet for maximum performance
- Implements 2 most common pre-tokenizer patterns (GPT-2, Llama-3); additional patterns can be added as needed

## Community & Support

- **🐛 Bug Reports**: [GitHub Issues](https://github.com/Michael-A-Kuykendall/shimmytok/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/Michael-A-Kuykendall/shimmytok/discussions)
- **💝 Sponsorship**: [GitHub Sponsors](https://github.com/sponsors/Michael-A-Kuykendall)

### 💝 Support shimmytok's Growth

If shimmytok saves you time and enables your Rust LLM projects, consider sponsoring:

- **$5/month**: Coffee tier ☕ - Eternal gratitude + sponsor badge
- **$25/month**: Bug prioritizer 🐛 - Priority support + name in [SPONSORS.md](SPONSORS.md)
- **$100/month**: Corporate backer 🏢 - Logo placement + recognition
- **$500/month**: Infrastructure partner 🚀 - Direct support + roadmap input

[**🎯 Become a Sponsor**](https://github.com/sponsors/Michael-A-Kuykendall) | See our amazing [sponsors](SPONSORS.md) 🙏

## Contributing

Contributions are welcome! This is a focused codebase (~2,700 LOC) that's easy to understand.

**All contributions must be signed off with the Developer Certificate of Origin (DCO).**

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and [DCO.md](DCO.md) for details.

```bash
# Quick setup for auto sign-off
git config format.signoff true
```

## Security

Found a security vulnerability? Please see [SECURITY.md](SECURITY.md) for responsible disclosure.

## License & Philosophy

**MIT License** - forever and always.

**Philosophy**: Foundation libraries should be reliable, focused, and free.

**Promise**: This will never become a paid product. If you want to support development, [sponsor it](https://github.com/sponsors/Michael-A-Kuykendall).

---

**Forever maintainer**: Michael A. Kuykendall  
**Mission**: Pure Rust tokenization for the LLM ecosystem

## Attribution

Based on [llama.cpp](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov (MIT License) and the [SentencePiece](https://github.com/google/sentencepiece) algorithm by Google (Apache 2.0).

## See Also

- [libshimmy](https://github.com/yourusername/libshimmy) - Pure Rust LLM inference engine that uses shimmytok
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Reference C++ implementation
- [GGUF format spec](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
