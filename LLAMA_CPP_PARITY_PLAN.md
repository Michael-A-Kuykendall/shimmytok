# llama.cpp Parity Implementation Plan

**Status**: ✅ COMPLETE  
**Scope**: Tier 1 fixes (llama.cpp behavioral parity for SPM + BPE)  
**Estimate**: 21 Fibonacci points total  
**Tests**: 13 passing in `tests/test_llama_cpp_parity.rs`

---

## Overview

This plan addresses gaps identified in `TOKENIZER_COMPARE_LLAMA_CPP_vs_SHIMMYTOK.md` that can be implemented now without new algorithm ports.

---

## Task 1: Byte-token detokenization (3 pts) ✅

**Gap**: SPM decode doesn't convert `<0x0A>` → `\n`

**Current behavior** ([src/sentencepiece.rs](src/sentencepiece.rs)):
```rust
// decode() just concatenates token texts and replaces ▁ with space
// Byte tokens like <0x0A> are emitted as literal strings
```

**Required behavior**:
- Detect tokens with `TokenType::Byte` or tokens matching `<0x[0-9A-Fa-f]{2}>`
- Emit the actual byte value instead of the token string

**Implementation**:
- [x] Add `fn decode_byte_token(text: &str) -> Option<u8>` helper
- [x] In `SentencePieceTokenizer::decode()`, check token type/text before appending
- [x] If byte token, append the raw byte to output buffer
- [x] Add test: encode "hello\nworld", decode, verify newline preserved

---

## Task 2: Honor `add_space_prefix` flag (3 pts) ✅

**Gap**: SPM always prefixes `▁` regardless of GGUF flag

**Current behavior** ([src/sentencepiece.rs](src/sentencepiece.rs)):
```rust
let processed_text = if text.starts_with(' ') {
    text.replace(' ', "▁")
} else {
    format!("▁{}", text.replace(' ', "▁"))  // Always adds prefix
};
```

**Required behavior**:
- Consult `vocab.add_space_prefix()` before adding leading `▁`
- llama.cpp also conditions on "previous fragment was special" — defer that for now

**Implementation**:
- [x] Pass `add_space_prefix` from vocab into SPM encode logic
- [x] Only prefix `▁` when flag is true AND text doesn't start with space
- [x] Add test with a model where `add_space_prefix = false`

---

## Task 3: `parse_special` mode (5 pts) ✅

**Gap**: No way to tokenize embedded special tokens like `<|eot_id|>`

**Current behavior**:
- All input treated as plain text
- Special tokens only added via `add_special_tokens` flag (BOS/EOS)

**Required behavior**:
- When `parse_special=true`, scan input for known special token strings
- Emit those as their token IDs directly (no space prefix, no merging)
- Tokenize the gaps between them normally

**Implementation**:
- [x] Build `HashMap<&str, TokenId>` of special token strings → IDs at load time
- [x] Add `encode_with_options(text, opts: EncodeOptions)` or modify `encode` signature
- [x] Pre-pass: find all special token occurrences, split text around them
- [x] Emit special token IDs for matches, normal tokenization for gaps
- [x] Add test: `"Hello<|eot_id|>World"` with parse_special=true

**API addition**:
```rust
pub struct EncodeOptions {
    pub add_special_tokens: bool,  // existing behavior
    pub parse_special: bool,       // NEW: detect special strings in input
}
```

---

## Task 4: Track additional special token IDs (3 pts) ✅

**Gap**: Only `bos`, `eos`, `unk`, `pad` tracked; llama.cpp exposes more

**Required tokens** (from GGUF `tokenizer.ggml.*`):
- `eot_token_id` (end of turn)
- `eog_token_id` (end of generation)  
- `sep_token_id` (separator)
- `nl_token_id` (newline)
- `fim_pre_token_id`, `fim_suf_token_id`, `fim_mid_token_id` (fill-in-middle)
- `mask_token_id`

**Implementation**:
- [x] Add optional fields to `Vocabulary` struct
- [x] Read from GGUF metadata in `vocab.rs`
- [x] Add public accessors: `eot_token()`, `sep_token()`, etc.
- [x] Update `is_special_token()` to include these

**Files to modify**:
- [src/vocab.rs](src/vocab.rs): Add fields + accessors
- [src/gguf.rs](src/gguf.rs): Read the new metadata keys

---

## Task 5: Token-attribute filtering in decode (5 pts) ✅

**Gap**: `decode` only has `skip_special_tokens: bool`; llama.cpp has more nuance

**llama.cpp behavior**:
- `token_to_piece(token, lstrip, special)`:
  - `lstrip`: strip leading whitespace from piece
  - `special`: if false, return empty for control/special tokens
- `detokenize(..., remove_special, unparse_special)`:
  - More granular control over special token handling

**Implementation**:
- [x] Add `DecodeOptions` struct:
  ```rust
  pub struct DecodeOptions {
      pub skip_special_tokens: bool,  // existing
      pub lstrip: bool,               // NEW
      pub include_special_text: bool, // NEW: emit special token text or empty
  }
  ```
- [x] Add `decode_with_options(tokens, opts)` method
- [x] Modify `token_to_piece` to accept optional `lstrip` param
- [x] Keep existing `decode(tokens, skip_special)` as convenience wrapper

---

## Task 6: Load cleanup/normalization flags (2 pts) ✅

**Gap**: GGUF flags like `clean_spaces`, `remove_extra_whitespaces` not loaded

**Implementation**:
- [x] Add fields to `Vocabulary`:
  ```rust
  clean_spaces: bool,
  remove_extra_whitespaces: bool,
  escape_whitespaces: bool,
  treat_whitespace_as_suffix: bool,
  ```
- [x] Read from GGUF metadata
- [x] Add public accessors
- [x] Wire into encode/decode (separate task, model-dependent)

**Note**: Loading is easy; full behavioral implementation is model-dependent and may need per-model validation.

---

## Acceptance Criteria

All tasks complete when:

1. `cargo test` passes
2. `cargo clippy` has zero warnings
3. Each task has at least one test exercising the new behavior
4. Existing tests still pass (no regressions)
5. README updated to document new options (if public API changed)

---

## Files to Modify

| File | Tasks |
|------|-------|
| [src/sentencepiece.rs](src/sentencepiece.rs) | 1, 2 |
| [src/vocab.rs](src/vocab.rs) | 3, 4, 6 |
| [src/gguf.rs](src/gguf.rs) | 4, 6 |
| [src/lib.rs](src/lib.rs) | 3, 5 (API additions) |
| [src/bpe.rs](src/bpe.rs) | 3 (if BPE needs parse_special too) |
| tests/ | All tasks need tests |

---

## Order of Implementation

1. **Task 4** (special token IDs) — foundation, no deps
2. **Task 6** (load flags) — foundation, no deps  
3. **Task 1** (byte-token decode) — self-contained
4. **Task 2** (add_space_prefix) — self-contained
5. **Task 3** (parse_special) — depends on Task 4
6. **Task 5** (decode options) — depends on Tasks 1, 4

---

## Out of Scope (for this plan)

- WPM (WordPiece) algorithm implementation
- UGM (Unigram/T5) algorithm implementation
- RWKV (greedy) algorithm implementation
- PLAMO2 (Aho-Corasick + DP) algorithm implementation

These require separate algorithm ports. See `ALGORITHM_PORTS_PLAN.md` (to be created from external work).
