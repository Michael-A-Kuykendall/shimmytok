# Shimmytok Tokenization Refactor Plan

**Date:** 2025-12-04  
**Context:** libshimmy’s TinyLlama stack now uses an internal SentencePiece-from-GGUF tokenizer (`SpmTokenizer`) that has been logic-audited against llama.cpp. Shimmytok was an earlier, more generic tokenizer crate; this document explains how tokenization is done *correctly* in libshimmy today and how shimmytok should be refactored to match and exceed that behavior.

---

## 1. How Tokenization Works in the Current libshimmy Stack

### 1.1 Source of Truth: GGUF Tokenizer Metadata

For TinyLlama and similar LLaMA-family models, libshimmy treats the GGUF file as the **single source of truth** for tokenization:

- Vocab strings: `tokenizer.ggml.tokens`
- Unigram scores (log-probs): `tokenizer.ggml.scores`
- Token types: `tokenizer.ggml.token_type` (normal, user-defined, control, unknown, etc.)
- Special-token IDs: BOS, EOS, UNK, etc. (dedicated GGUF metadata fields)
- Model type: SentencePiece unigram (for TinyLlama and LLaMA-style models)

libshimmy does **not** hardcode vocab entries, scores, token IDs, or special-token IDs. Everything is read from GGUF at runtime.

### 1.2 Internal SPM Tokenizer (libshimmy)

The production tokenizer in libshimmy is `SpmTokenizer` (`src/tokenizer/spm_tokenizer.rs`). At a high level:

- **Algorithm:** Viterbi over a SentencePiece unigram model.
- **Preprocessing:**
  - Prepend a leading space to the input (matching llama.cpp’s SPM path).
  - Convert spaces to U+2581 (`▁`, ESCAPED_SPACE), as in standard SentencePiece.
- **State representation:**
  - Dynamic programming (DP) table indexed by byte offset.
  - Each DP cell stores: best cumulative score + backpointer to previous token.
- **Scoring:**
  - For each candidate token at a position, add its log-probability from GGUF (`score`).
  - User-defined tokens (type 4) are given score `0.0` (neutral) to allow them to compete correctly.
  - Unknown fragments use a fixed low log-prob (`min_score - 10.0`), and consecutive unknowns are merged.
- **Decoding:**
  - Walk backpointers from the best final state to build the token sequence.
- **Special tokens:**
  - BOS/EOS/UNK IDs are taken directly from GGUF metadata.
  - `encode(text, add_bos)` prepends BOS when requested using the GGUF-provided BOS ID.
  - `decode(tokens)` maps IDs back to strings using the GGUF vocab.

### 1.3 Parity with llama.cpp

libshimmy’s SPM tokenizer behavior has been directly compared with llama.cpp’s `llm_tokenizer_ugm_session::tokenize` via:

- A small example test for phrases like `"Hello"` and `"Hello world"` with and without BOS/EOS.
- A parity example that calls `llama-tokenize` from llama.cpp and compares token ID sequences.

The internal SPM implementation is therefore treated as **canonical** for TinyLlama in libshimmy.

### 1.4 Additional Tokenizer Paths in libshimmy

Beyond `SpmTokenizer`, libshimmy also exposes:

- `DirectGgufTokenizer` (`src/tokenizer/direct_gguf.rs`):
  - Simple tokenizer that uses the raw GGUF vocab without full SPM; useful for debugging and experiments.
- `TiktokenWrapper` (`src/tokenizer/tiktoken_wrapper.rs`):
  - Integrates tiktoken for experimentation with BPE models or alternative tokenization strategies.

These are **non-production** for TinyLlama parity but important as reference designs.

---

## 2. How Shimmytok Originally Did Tokenization (High-Level)

Shimmytok’s original design goals (based on prior audits and docs):

- Provide a **crate-level tokenizer API** that libshimmy and other clients could depend on.
- Support both **SentencePiece unigram** and **BPE** tokenization using GGUF metadata.
- Offer a simple, unified API:
  - `Tokenizer::from_gguf_file(path)`
  - `encode(text, add_special_tokens)` / `decode(tokens, skip_special_tokens)`
  - `bos_token()`, `eos_token()`, `vocab_size()`, etc.

However, the earlier shimmytok implementation had several problems in practice:

- **Mismatch with GGUF SPM behavior:**
  - Different pre/post-processing compared to llama.cpp (e.g., space handling, U+2581, unknown token treatment).
  - Possibly different scoring/selection rules for SentencePiece unigram.
- **Hardcoded or assumed BOS/EOS IDs:**
  - Some integration paths assumed BOS=1 and EOS=2 instead of reading from GGUF.
- **Incomplete or experimental BPE support:**
  - BPE was stubbed for some model families and not fully validated against reference tokenizers.

These gaps are why libshimmy eventually moved to an **internal, directly-audited SPM implementation** instead of relying on shimmytok.

---

## 3. What libshimmy Got Right (vs. Earlier Shimmytok)

These are the key correctness properties of libshimmy’s current tokenization that shimmytok should adopt:

1. **GGUF-as-Oracle:**
   - All tokenizer behavior (vocab, scores, token types, special IDs, model type) is read from GGUF.
   - No hardcoded BOS/EOS IDs or assumptions about model family.

2. **Exact SPM Unigram Semantics:**
   - Viterbi DP over byte offsets with backpointers.
   - Correct U+2581 handling and leading space canonicalization.
   - User-defined tokens treated with neutral score (0.0) so they are selectable.
   - Unknown tokens assigned a fixed low log-prob and merged when consecutive.

3. **Per-Model Special Tokens:**
   - `bos_token()`, `eos_token()`, and `unk_token()` returned directly from GGUF metadata.
   - `encode(text, add_bos)` respects those IDs and does not guess.

4. **Parity-First Mindset:**
   - All behavior is validated against llama.cpp (`llama-tokenize`) via small, explicit parity tests.
   - TinyLlama SPM behavior is treated as the ground truth.

5. **Clear Separation of Concerns:**
   - Internal implementation (`SpmTokenizer`) is hidden behind a small enum/API in libshimmy (`Tokenizer`), making it easy to swap or evolve.

---

## 4. Refactor Plan for Shimmytok (Going Forward)

Shimmytok can be rehabilitated and made genuinely useful by **reusing** the audited logic from libshimmy and reshaping its API. The plan below assumes you will do this work later; this document is the blueprint.

### 4.1 High-Level Goals

1. **Make shimmytok a thin, reusable tokenizer library** whose behavior matches llama.cpp and other reference implementations for supported model families.
2. **Share logic with libshimmy** where possible so there is a single source of truth for SPM-from-GGUF and, eventually, BPE-from-GGUF.
3. **Preserve and extend the public API** so shimmytok is attractive as a standalone crate (e.g. for other inference engines or tools).

### 4.2 Concrete Refactor Steps

#### Step 1: Introduce a Model-Agnostic Tokenizer Core

In shimmytok, define an internal core that is agnostic to libshimmy but aligned with its logic:

- `enum TokenizerKind { SpmUnigram, Bpe }`
- `struct TokenizerConfig { model_type, special_ids, vocab, scores, merges (for BPE), token_types, ... }`
- A builder that **only** takes GGUF metadata as input (no hardcoded values):
  - `TokenizerConfig::from_gguf(path: impl AsRef<Path>) -> Result<TokenizerConfig>`

This mirrors what libshimmy currently does, but in a crate that doesn’t depend on libshimmy itself.

#### Step 2: Port/Reuse libshimmy’s SPM Implementation

Either:

- **Option A (recommended):** Extract the core SPM logic from libshimmy’s `SpmTokenizer` into a small shared crate (e.g. `shimmy-spm-core`) and have both libshimmy and shimmytok depend on it; or
- **Option B:** Carefully re-implement the same algorithm in shimmytok, line-by-line matching libshimmy’s audited behavior.

Key points to copy exactly:

- DP structure (states indexed by byte offset).
- Space handling and U+2581 transforms.
- Score usage from GGUF scores array.
- Unknown-token fallback and merging.
- Behavior for user-defined tokens and other token types.

Once this is done, shimmytok’s `TokenizerKind::SpmUnigram` should:

- Read GGUF metadata.
- Build the internal SPM model.
- Provide `encode` / `decode` / `bos_token` / `eos_token` consistent with libshimmy.

#### Step 3: Make Special Tokens Fully Data-Driven

In shimmytok’s public API:

- Replace any hardcoded BOS/EOS IDs with lookups from `TokenizerConfig`.
- Provide:
  - `fn bos_token(&self) -> Option<u32>`
  - `fn eos_token(&self) -> Option<u32>`
  - `fn unk_token(&self) -> Option<u32>`
- Ensure `encode(text, add_bos)` and `decode(tokens, skip_special_tokens)` use these values, not magic numbers.

#### Step 4: Add a Parity Test Harness (Using libshimmy’s Pattern)

Create a small `examples/` directory in shimmytok mirroring libshimmy’s tests:

- `examples/test_spm_tinyllama_parity.rs`:
  - Load TinyLlama GGUF via shimmytok.
  - Tokenize a handful of strings (`"Hello"`, `"Hello world"`, prompts with newlines).
  - Optionally call `llama-tokenize` and compare ID sequences.
- `examples/test_roundtrip.rs`:
  - Encode/decode roundtrip tests for a few strings.

This acts as a **guardrail** to keep shimmytok behavior in sync with libshimmy and llama.cpp.

#### Step 5: Decide What to Do with BPE

Shimmytok originally had aspirations to support generic BPE (e.g. Qwen, Phi-2). For that path:

- Keep BPE support **explicitly experimental** until:
  - You have a GGUF-based BPE implementation that matches a reference tokenizer (e.g. Qwen2, Phi-2) on test cases.
  - You have at least one parity example similar to the SPM tests.
- Clearly document BPE status in shimmytok’s README as:
  - `SentencePiece (LLaMA/TinyLlama-style): PRODUCTION-READY (matches llama.cpp)`
  - `BPE (Qwen/Phi-2/...): EXPERIMENTAL until parity is proven`

#### Step 6: Simplify the Public API

A cleaned-up shimmytok API might look like:

```rust
pub enum ModelType {
    SentencePieceUnigram,
    Bpe,
}

pub struct Tokenizer {
    kind: ModelType,
    // internal fields
}

impl Tokenizer {
    pub fn from_gguf_file(path: impl AsRef<Path>) -> Result<Self> { /* ... */ }

    pub fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> { /* ... */ }
    pub fn decode(&self, tokens: &[u32], skip_special: bool) -> Result<String> { /* ... */ }

    pub fn bos_token(&self) -> Option<u32> { /* ... */ }
    pub fn eos_token(&self) -> Option<u32> { /* ... */ }
    pub fn vocab_size(&self) -> usize { /* ... */ }
}
```

Internally, the SentencePiece branch should be as close as possible to libshimmy’s current `SpmTokenizer` implementation.

---

## 5. Recommendations Summary (For Future You)

When you come back to shimmytok, treat this as the checklist:

1. **Align behavior with libshimmy’s SPM tokenizer**:
   - Either vendor the core algorithm or factor it into a shared crate.
   - Use GGUF metadata exclusively for vocab, scores, and special tokens.

2. **Kill magic numbers:**
   - Ensure **no** BOS/EOS IDs are hardcoded; always read from GGUF.

3. **Add parity tests:**
   - Reuse libshimmy’s test patterns to compare shimmytok tokens against llama.cpp for TinyLlama.

4. **Document the truth:**
   - In shimmytok’s README, clearly state:
     - SPM support is aligned with libshimmy and llama.cpp.
     - BPE support is experimental until parity is proven.

5. **Keep libshimmy and shimmytok decoupled but consistent:**
   - libshimmy’s internal tokenizer remains the production path for TinyLlama.
   - shimmytok becomes a **general-purpose tokenizer crate** that happens to share the same audited logic.

If you follow this plan, shimmytok will stop being a potential source of confusion and become a clean, reusable tokenizer library whose behavior you can trust, while libshimmy continues to rely on its own audited SPM implementation for TinyLlama parity.
