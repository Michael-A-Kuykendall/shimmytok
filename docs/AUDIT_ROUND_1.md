# Hostile Audit Round 1

## Audit Date: 2025-10-21

### Methodology
Deep audit of every file in src/ and tests/, looking for:
- Logic errors that would cause incorrect tokenization
- Edge cases not handled
- Unsafe assumptions
- Missing validation
- Performance issues
- API design flaws
- Test coverage gaps

---

## CRITICAL ISSUES

### 1. BPE decode is WRONG - Missing byte reversal validation
**File**: `src/bpe.rs:263-270`
**Severity**: CRITICAL

```rust
pub fn decode(&self, tokens: &[TokenId], vocab: &Vocabulary) -> String {
    let byte_encoded_text: String = tokens
        .iter()
        .filter_map(|&id| vocab.get_token_text(id))
        .collect::<Vec<_>>()
        .join("");
    
    crate::byte_encoder::decode_bytes(&byte_encoded_text)
}
```

**Problem**: What if `decode_bytes()` returns invalid UTF-8? We're blindly calling `into_owned()` on a lossy conversion. This will silently corrupt data.

**Impact**: Silent data corruption on invalid sequences.

**Fix**: Return Result<String, Error> and handle UTF-8 validation explicitly.

---

### 2. BPE merge algorithm has NO VALIDATION of merge rank bounds
**File**: `src/bpe.rs:133-207`
**Severity**: CRITICAL

The entire merge loop assumes merge_ranks HashMap contains valid data. What if:
- A merge rank is corrupted?
- merge_ranks contains a (left, right) pair that doesn't exist in vocab?
- The merge produces a token that's not in vocab?

**Problem**: Zero validation that the merged token exists in vocabulary.

```rust
if let Some(&rank) = merge_ranks.get(&(left_text.to_string(), right_text.to_string())) {
    work_queue.push(Bigram {
        left,
        right,
        rank,
        text: format!("{}{}", left_text, right_text),
    });
}
```

Then later:
```rust
let merged_token = vocab.token_to_id(&work_queue.peek().unwrap().text);
```

**What if this returns None?** We call `unwrap()` on line 178 - PANIC!

**Impact**: Panics on corrupted GGUF files.

**Fix**: Validate merged token exists, return Error if not.

---

### 3. SentencePiece encode has UNBOUNDED RECURSION RISK
**File**: `src/sentencepiece.rs:80-132`
**Severity**: HIGH

```rust
fn try_merge(&self, symbols: &mut Vec<(usize, usize)>, vocab: &Vocabulary) -> bool {
    // ... finds best merge
    if let Some((best_pos, _best_score, merged_token_id)) = best_merge {
        // Merge the symbols
        symbols[best_pos].1 = symbols[best_pos + 1].1;
        symbols.remove(best_pos + 1);
        true
    } else {
        false
    }
}
```

Then in encode:
```rust
while self.try_merge(&mut symbols, vocab) {}
```

**Problem**: If `try_merge` has a bug and always returns true, this is infinite loop. No iteration limit!

**Impact**: Infinite loop = hang = DoS.

**Fix**: Add max iteration count (e.g., 10 * text.len()).

---

### 4. Vocabulary has NO SIZE LIMITS
**File**: `src/vocab.rs`
**Severity**: MEDIUM

```rust
pub fn token_to_id(&self, token: &str) -> Option<TokenId> {
    self.token_to_id.get(token).copied()
}
```

A malicious GGUF file could have:
- 10 million tokens
- Tokens with 10MB strings
- Duplicate token IDs

**Problem**: No validation of:
- vocab.len() < some reasonable max (100k?)
- token text length < some max (1kb?)
- All token IDs are unique
- Token IDs are sequential from 0

**Impact**: Memory exhaustion attacks.

**Fix**: Add validation in from_gguf_file.

---

### 5. GGUF parsing accepts UNTRUSTED INPUT with no size limits
**File**: `src/gguf.rs:164-191`
**Severity**: HIGH

```rust
fn read_string(reader: &mut impl Read) -> Result<String, Error> {
    let len = read_u64(reader)?;
    let mut buf = vec![0u8; len as usize];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| Error::InvalidMetadata(format!("Invalid UTF-8: {}", e)))
}
```

**Problem**: `len` could be 4GB! This allocates `vec![0u8; 4GB]` - instant OOM.

**Impact**: Trivial DoS via malicious GGUF.

**Fix**: Add MAX_STRING_SIZE = 1MB, reject larger.

---

### 6. Error handling is INCONSISTENT - panics vs Results
**File**: Multiple

- `vocab.rs:140` - `expect()` on token lookup (PANIC)
- `bpe.rs:178` - `unwrap()` on merged token (PANIC)  
- `sentencepiece.rs:115` - Silent fallback on missing token

**Problem**: API claims to return Result, but actually panics in many code paths.

**Impact**: Impossible to write robust code using this library.

**Fix**: NEVER panic in library code. Return Err() always.

---

### 7. BPE pre_tokenize can FAIL but is unwrapped
**File**: `src/bpe.rs:247`

```rust
let fragments = self.pre_tokenize(&text_encoded, vocab)
    .expect("BPE pre-tokenization regex failed - this should never happen with hardcoded patterns");
```

**Problem**: "Should never happen" is famous last words. Regex could:
- Fail on deeply nested input
- Fail on very long input (catastrophic backtracking)
- Fail on malformed unicode

**Impact**: Panic on adversarial input.

**Fix**: Return Result, don't expect().

---

### 8. Tests use HARDCODED paths that WILL BREAK in CI
**File**: `tests/test_bpe.rs:4-9`

```rust
fn get_model_path() -> String {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(|home| format!("{}/.cache/models/gguf/gpt2.Q4_K_M.gguf", home))
        .unwrap_or_else(|_| "gpt2.Q4_K_M.gguf".to_string())
}
```

**Problem**: 
- CI won't have ~/.cache/models/gguf/gpt2.Q4_K_M.gguf
- Tests will fail in fresh checkout
- No documentation on how to set up test fixtures

**Impact**: Tests don't run in CI = broken main branch.

**Fix**: 
- Include test fixture in repo (small vocab GGUF)
- Or use env var SHIMMYTOK_TEST_MODEL
- Or skip test if model not found

---

### 9. byte_encoder has UNBOUNDED HashMap growth
**File**: `src/byte_encoder.rs:10-42`

```rust
pub fn bytes_to_unicode() -> &'static HashMap<u8, char> {
    static BYTE_ENCODER: OnceLock<HashMap<u8, char>> = OnceLock::new();
    BYTE_ENCODER.get_or_init(|| {
        // ... builds 256-entry hashmap
    })
}
```

**Problem**: This is fine, but decode_bytes() could be called with GB of data, no streaming.

```rust
pub fn decode_bytes(text: &str) -> String {
    let byte_decoder = unicode_to_bytes();
    let bytes: Vec<u8> = text.chars()
        .filter_map(|c| byte_decoder.get(&c).copied())
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}
```

**Impact**: Memory spike = OOM on large documents.

**Fix**: Not urgent, but document max input size or make streaming.

---

### 10. No input validation on public API
**File**: `src/lib.rs:134-148`

```rust
pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<TokenId>, Error> {
    let mut tokens = Vec::new();
    
    if add_special_tokens && self.vocab.add_bos_token() {
        tokens.push(self.vocab.bos_token_id());
    }
    
    tokens.extend(self.tokenizer_impl.encode(text, &self.vocab));
    // ...
}
```

**Problems**:
- No max text length check (could pass 10GB string)
- No validation that result fits in Vec (could have 100M tokens)
- No timeout/limit on processing time

**Impact**: Trivial DoS.

**Fix**: Add MAX_INPUT_SIZE constant, validate inputs.

---

### 11. GGUF version check is TOO LOOSE
**File**: `src/gguf.rs:39-41`

```rust
if !(2..=3).contains(&version) {
    return Err(Error::InvalidMetadata(format!("Unsupported GGUF version: {}", version)));
}
```

**Problem**: Version 3 might have different format than version 2. Do we actually support both?

**Impact**: Could accept GGUF v3 files we can't parse correctly.

**Fix**: Test with both versions or only accept version we've tested (probably 2).

---

### 12. Merge rules are loaded into UNBOUNDED HashMap
**File**: `src/vocab.rs:94-107`

```rust
pub fn load_merge_rules(&mut self, merges: Vec<String>) {
    for (rank, merge_text) in merges.iter().enumerate() {
        let parts: Vec<&str> = merge_text.split(' ').collect();
        if parts.len() == 2 {
            self.merge_ranks.insert(
                (parts[0].to_string(), parts[1].to_string()),
                rank,
            );
        }
    }
}
```

**Problems**:
- No validation that parts[0] and parts[1] are valid tokens in vocab
- No validation of merge count (could be 10M merges)
- No handling of duplicate merges

**Impact**: Accept invalid GGUF, waste memory.

**Fix**: Validate merges reference actual tokens.

---

## MEDIUM ISSUES

### 13. GPT2_PATTERN and LLAMA3_PATTERN might not match reference implementations
**File**: `src/bpe.rs:12-15`

Comment says "Simplified from original" - this is RED FLAG.

**Problem**: If regex differs from OpenAI's GPT-2, tokenization will be WRONG.

**Fix**: Need validation tests against known-good GPT-2 outputs.

---

### 14. Special token handling is unclear
**File**: `src/vocab.rs`

```rust
pub fn bos_token_id(&self) -> TokenId {
    self.bos_token
}
```

**Problem**: What if GGUF doesn't have BOS token? This returns 0 (default).

**Impact**: Silently wrong tokenization if BOS is actually token 0 vs "no BOS".

**Fix**: Use Option<TokenId> for special tokens.

---

### 15. No #[must_use] on Result types
**File**: All

**Problem**: Easy to ignore errors:
```rust
tokenizer.encode(text, false); // forgot .unwrap()!
```

**Fix**: Add #[must_use] to Result-returning functions.

---

## TEST COVERAGE GAPS

### 16. No tests for error paths
- What happens if GGUF file is truncated?
- What if vocab is empty?
- What if merge rules reference non-existent tokens?
- What if token ID > vocab size?

**Fix**: Add negative tests in tests/test_errors.rs

---

### 17. No tests for large inputs
- 1MB string
- 100k tokens
- Very long tokens (1kb)

**Fix**: Add tests/test_large_inputs.rs

---

### 18. No concurrency tests
- Is Tokenizer Send + Sync?
- Can we use from multiple threads?
- Are there any race conditions?

**Fix**: Add tests/test_concurrent.rs

---

### 19. Edge case: empty vocab
What if GGUF has 0 tokens?

**Fix**: Add test + validation.

---

### 20. Edge case: token ID overflow
What if GGUF claims 5B tokens? (u32::MAX)

**Fix**: Validate n_tokens < reasonable max.

---

## Summary

**Critical Issues**: 7 (could cause crashes/corruption)
**High Issues**: 3 (DoS/security)
**Medium Issues**: 10 (correctness/robustness)
**Total Issues Found**: 20

**Next Step**: Fix all CRITICAL issues first.
