# API Stability Contract

This document defines which parts of the shimmytok public API are stable,
what downstream code (primarily airframe) depends on, and the rules for how
each item may change.

---

## Stability levels

| Level | Meaning |
|-------|---------|
| **Stable** | No breaking changes without a semver major bump |
| **Committed** | Stable in all 0.7.x releases; may change at 0.8.0 with notice |
| **Experimental** | May change in any release; documented as `⚠️ Experimental` |

---

## What airframe depends on (as of v0.7.2)

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

### Types used

| Type | Status |
|------|--------|
| `Tokenizer` | **Stable** — opaque struct, `Send + Sync` |
| `TokenId` (`u32`) | **Stable** — type alias, will not change underlying type |
| `EncodeOptions` | **Stable** — fields are public but construct via the named constructors |
| `Error` | **Committed** — `#[non_exhaustive]`; always match with a `_` arm |

---

## What is NOT used by airframe (safe to change internally)

- `DecodeOptions`
- `decode` / `decode_with_options`
- `encode_batch`
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

### What WILL break at a major version bump (1.0.0, 2.0.0, …)

- Removing or renaming any stable method.
- Changing return types.
- Making `Tokenizer` non-`Send` or non-`Sync`.

---

## Updating airframe

To upgrade the pinned version:

```toml
# airframe/Cargo.toml
shimmytok = "0.7.2"   # or "0.7" to track the latest 0.7.x automatically
```

Any `0.7.x` release is safe to pull without code changes in airframe.
A `0.8.0` release will be accompanied by a migration note here and in CHANGELOG.

---

## Verification

To confirm no breaking changes were introduced between versions, run:

```bash
# In the shimmytok repo
cargo test --quiet

# In the airframe repo (after bumping the version)
cargo build --quiet
```

Both must pass clean before tagging a release.
