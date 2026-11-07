# API Stability Contract — ShimmyTok 0.8.5

This document defines which parts of the shimmytok public API are stable,
what downstream code (primarily airframe) depends on, and the rules for how
each item may change.

---

## Stability levels

| Level | Meaning |
|-------|---------|
| **Stable** | No breaking changes without a semver major bump |
| **Committed** | Stable within the 0.8.x series; compatibility changes require a versioned changelog notice |
| **Experimental** | May change in any release; documented as `⚠️ Experimental` |

---

## Airframe 0.4.5 integration with ShimmyTok 0.8.5

Determined by scanning `airframe/src` for all `shimmytok::` usages.

### Imports used

```rust
use shimmytok::Tokenizer;
use shimmytok::{EncodeOptions, Tokenizer};
```

### Methods called

| Method | Signature | Status |
|--------|-----------|--------|
| `Tokenizer::from_gguf_file` | `(path: impl AsRef<Path>) -> Result<Tokenizer, Error>` | **Stable** |
| `tokenizer.encode` | `(text: &str, add_special: bool) -> Result<Vec<TokenId>, Error>` | **Stable** |
| `tokenizer.encode_with_options` | `(text: &str, opts: &EncodeOptions) -> Result<Vec<TokenId>, Error>` | **Stable** |
| `tokenizer.decode_single` | `(token: TokenId, skip_special: bool) -> Result<String, Error>` | **Stable** |
| `tokenizer.eos_token` | `() -> TokenId` | **Stable** |
| `EncodeOptions::with_parse_special` | `(add_special: bool, parse_special: bool) -> EncodeOptions` | **Stable** |
| `Tokenizer::encode_with_external_special_tokens` | `(text, options, overrides) -> Result<Vec<TokenId>, Error>` | **Committed** |
| `Tokenizer::metadata` | `() -> &HashMap<String, GGUFValue>` | **Committed** |

### Types used

| Type | Status |
|------|--------|
| `Tokenizer` | **Stable** — opaque struct, `Send + Sync` (verified by a compile-time assertion in the test suite) |
| `TokenId` (`u32`) | **Stable** — type alias, will not change underlying type |
| `EncodeOptions` | **Stable** — fields are public but construct via the named constructors |
| `SpecialTokenOverride` | **Committed** — caller-owned marker-to-token mapping for opt-in encoding |
| `GGUFValue` | **Committed** — typed, read-only representation of preserved GGUF metadata |
| `Error` | **Committed** — `#[non_exhaustive]`; always match with a `_` arm |

---

## Batch and lookup contracts (Committed as of 0.8.0)

These methods were previously internal-only. As of 0.8.0 they carry the
**Committed** guarantee: their behaviour is fixed for the 0.8.x series and will
only change at 0.9.0 with a CHANGELOG note.

| Method | Signature | Status | Contract |
|--------|-----------|--------|----------|
| `tokenizer.encode_batch` | `(texts: &[&str], add_special: bool) -> Result<Vec<Vec<TokenId>>, Error>` | **Committed** | Available in the public API with sequential or optional parallel execution; target builds not executed in this release record are NOT_RUN. Output order matches input order. Each element equals the corresponding single `encode` call. On multiple failures, returns the error at the **lowest failing input index**, identical across sequential and parallel backends. |
| `tokenizer.get_token` | `(text: &str) -> Option<TokenId>` | **Committed** | Exact-match lookup only. No normalization, alternate-space handling, or special parsing. Returns the vocabulary ID for an exact token piece, or `None` if absent. |

`encode_batch` never exposes Rayon (or any parallelism library) in its
signature. Whether a batch runs in parallel is an internal, measured decision
controlled by the `parallel` feature and a data-backed size threshold.

---

## External Special-Token Contract

`Tokenizer::encode_with_external_special_tokens` is an additive, per-call API.
It preserves the existing `EncodeOptions` layout and does not mutate shared
`Tokenizer` state.

- External markers are parsed only when `EncodeOptions::parse_special` is true.
- Marker IDs must be less than `Tokenizer::vocab_size()`.
- Empty marker text, duplicate marker text, and collisions with built-in special
  markers return `Error::InvalidSpecialToken`.
- Marker order follows the input text; BOS/EOS behavior remains controlled only
  by `EncodeOptions::add_special_tokens` and the GGUF flags.
- `Tokenizer::metadata()` returns a read-only map of all parsed GGUF metadata,
  including model-specific scalar and array values represented by `GGUFValue`.

## Feature flags

| Feature | Default | Public? | Meaning |
|---------|---------|---------|---------|
| `parallel` | **on** | Internal (`#[doc(hidden)]`) | Enables the optional Rayon batch implementation; disabling it selects sequential code. API output-order/error contracts are unchanged, while unexecuted target configurations remain NOT_RUN. |

Consumers should not rely on the presence or absence of `parallel`; it is an
implementation detail. `encode_batch` remains available; output-order and
lowest-failing-index contracts are not a claim that every target was certified.

---

## Additional public surface

Not every consumer call is listed above. Being unused by Airframe is not
permission to break another consumer's API; review compatibility and tests.

- `DecodeOptions`
- `decode` / `decode_with_options`
- `bos_token`
- `vocab_size`
- `model_type` / `pre_type`
- `token_type` / `is_special_token` / `token_to_piece`
- `TokenType` enum
- `Vocabulary` struct (pub but internal detail)
- `invariants` module
- All tokenizer impl structs (`BPETokenizer`, `WpmTokenizer`, etc.)

---

## Rules for safe upgrades

### What will NEVER change without a major version bump

1. The `Tokenizer` struct remains `Send + Sync`.
2. `from_gguf_file` accepts `AsRef<Path>` and returns `Result<Tokenizer, Error>`.
3. `encode` / `encode_with_options` return `Vec<u32>` in the same order as tokens appear in text.
4. `decode_single` returns an empty `String` (not an error) when `skip_special = true` and the token is special.
5. `eos_token` returns a `u32`; the value matches the model's EOS from GGUF metadata.
6. `TokenId` is always `u32`.
7. `EncodeOptions::with_parse_special` signature unchanged.

### What MAY change at a minor version bump (0.8.0, 0.9.0, …)

- New variants on `Error` (already `#[non_exhaustive]` — use `_` arm).
- New fields on `EncodeOptions` / `DecodeOptions` (use named constructors, not struct literals).
- New methods added to `Tokenizer` (additive, not breaking).
- MSRV raised (announced in CHANGELOG with advance notice).

### Changes requiring a breaking-compatibility review

- Removing or renaming any stable method.
- Changing return types.
- Making `Tokenizer` non-`Send` or non-`Sync`.

---

## Updating Airframe for 0.8.5

Airframe 0.4.5 consumes the coordinated tokenizer:

```toml
# Airframe Cargo.toml
shimmytok = "0.8.5"
```

The 0.7.2 example and predictions about a future 0.8.0 migration are superseded.
The current release retains the existing tokenizer API and special-token
semantics. The new shared `manifest` module is the checked GGUF interpretation
used by Airframe's descriptor admission; it does not certify an execution graph.
See [GGUF_SECURITY.md](GGUF_SECURITY.md) for parser limits and reader regressions.

## Verification

```bash
cargo test --locked --test gguf_untrusted_lengths -- --test-threads=1
cargo test --locked -- --test-threads=1
```

Run these in the tokenizer source checkout. Consumer integration also needs the
engine's typed manifest/admission tests and exact model/reference comparisons.
Compiling successfully alone does not prove tokenizer parity, inference
correctness, platform support or release readiness. Unavailable fixtures and
targets stay NOT_RUN. Publication requires separate maintainer approval;
these local commands do not authorize a tag or upload.
