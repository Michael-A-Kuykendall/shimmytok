# ShimmyTok roadmap — 0.8.5

Current integration is ShimmyTok 0.8.5 with Airframe 0.4.5 and Shimmy 2.6.6.
The [README](README.md) describes implemented tokenizer dispatch; the
[API contract](docs/API_STABILITY.md) describes compatibility. This is not a
universal llama.cpp parity or platform-certification claim.

## Delivered in 0.8.5

- Shared checked GGUF metadata/tensor interpretation for tokenizer and engine.
- Permanent impossible-array, allocation-budget, overflow and truncation
  regressions through the public reader.
- Existing tokenizer and opt-in external-special-token APIs retained.
- Documentation distinguishes implemented algorithms from executed fixtures,
  and fixes the obsolete 0.7.x integration/future-release guidance.

The earlier 0.8/0.9/1.0 schedule is superseded. No next-version date,
performance ratio, universal safety guarantee or complete-model support is
promised here. Future compatibility work requires exact model hashes, token IDs,
reference versions and reproducible acceptance tests.

## Remaining boundaries

Tokenizer implementation support does not imply Airframe executor admission.
Neural-network operators, cache semantics and platform certification belong to
the engine. Further tokenizer families or normalization variants need their own
fixtures; an implementation name alone is not independent numerical evidence.

This release's package tests executed locally on Linux under WSL. WASM/WASI,
other platforms and declared minimum-toolchain validation are NOT_RUN in the
release record. The Cargo manifest still declares Rust 1.70; do not interpret
that declaration as a newly executed minimum-toolchain matrix.

Untrusted input is bounded where described in the
[parser security notes](docs/GGUF_SECURITY.md), not universally immune to resource
exhaustion. Tokenizer training and model inference remain outside this library.

## Collaboration and licensing

Follow the [email-first contribution policy](CONTRIBUTING.md).
Security reports use [private disclosure](SECURITY.md).
Code remains [MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE);
attribution remains in [AUTHORS.md](AUTHORS.md).
