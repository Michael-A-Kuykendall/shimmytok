# 🤝 Contributing to shimmytok

**Welcome to shimmytok!**

## Open Source, Not Open Contribution

shimmytok is **open source** but **not open contribution**.

- The code is freely available under the MIT license
- You can fork, modify, use, and learn from it without restriction
- **Pull requests are not accepted by default**
- All architectural, roadmap, and merge decisions are made by the project maintainer

This model keeps the project coherent, maintains clear ownership, and ensures consistent quality. It's the same approach used by SQLite and many infrastructure projects.

## How to Contribute

If you believe you can contribute meaningfully to shimmytok:

1. **Email the maintainer first**: [michaelallenkuykendall@gmail.com](mailto:michaelallenkuykendall@gmail.com)
2. Describe your background and proposed contribution
3. If there is alignment, a scoped collaboration may be discussed privately
4. Only after discussion will PRs be considered

**Unsolicited PRs will be closed without merge.** This isn't personal — it's how this project operates.

## What We Welcome (via email first)

- Bug reports with detailed reproduction steps (Issues are fine)
- Security vulnerability reports (please email directly)
- Tokenization accuracy issues with specific models
- Documentation improvements (discuss first)

## What We Handle Internally

- New tokenizer implementations
- API design decisions
- GGUF format support
- Performance optimizations
- llama.cpp parity work

## Bug Reports

Bug reports via GitHub Issues are welcome! Please include:
- Model file used (or GGUF source)
- Input text that caused the issue
- Expected vs actual token output
- shimmytok version and Rust version
- Comparison with llama.cpp output if possible

## Code Style (for reference)

If a contribution is discussed and approved:
- Rust 2021 edition with `cargo fmt` and `cargo clippy`
- Comprehensive error handling using `thiserror`
- All public APIs must have documentation with examples
- Tests with llama.cpp validation

## shimmytok Philosophy

Any accepted work must align with:
- **Pure Rust**: No C++ dependencies
- **GGUF-focused**: Direct loading from model files
- **llama.cpp compatible**: Match reference implementation
- **Correctness over performance**: Get it right first
- **Minimal dependencies**: Lightweight and focused
- **Free Forever**: No features that could lead to paid tiers

## Developer Certificate of Origin (DCO)

All contributions must be signed off with the Developer Certificate of Origin. See [DCO.md](DCO.md) for details.

```bash
git config format.signoff true  # Auto sign-off all commits
```

## Why This Model?

Building a reliable tokenizer requires tight control over correctness. This ensures:
- Perfect match with llama.cpp reference implementation
- No ownership disputes or governance overhead
- Quality control without committee delays
- Clear direction for the project's future

The code is open. The governance is centralized. This is intentional.

## Recognition

Helpful bug reports and community members are acknowledged in release notes.
If email collaboration leads to merged work, attribution will be given appropriately.

---

**Maintainer**: Michael A. Kuykendall
