# Hostile Audit Round 3

## Audit Date: 2025-10-22

### Context
- Round 1: Fixed 20 issues (7 critical, 3 high, 10 medium)
- Round 2: Fixed 11 issues (4 critical, 2 high, 5 medium)
- Total: 31 issues fixed

This round audits ALL the fixes and looks for NEW vulnerabilities introduced by changes.

### Methodology
- Audit error handling paths for logic errors
- Check for off-by-one errors in limits
- Look for integer overflow in size calculations
- Verify all allocations are bounded
- Check for panic paths that survived
- Look for race conditions if used concurrently

---

## CRITICAL ISSUES

### 1. MAX_OUTPUT_TOKENS check can be bypassed via fragmentation
**File**: `src/bpe.rs:268-279`
**Severity**: CRITICAL

```rust
for fragment in fragments {
    let tokens = self.bpe_fragment(&fragment, vocab)?;
    if result.len() + tokens.len() > MAX_OUTPUT_TOKENS {
        return Err(...);
    }
    result.extend(tokens);
}
```

**Problem**: Check AFTER computing tokens for fragment!
- Fragment produces 500K tokens
- result.len() = 600K
- Check: 600K + 500K > 1M → error
- BUT: We already allocated 500K tokens in bpe_fragment()!

**Impact**: Can allocate 1.5M tokens before detection (60% over limit)

**Fix**: Check BEFORE calling bpe_fragment, or limit fragment size

---

### 2. Total string bytes tracking has integer overflow
**File**: `src/gguf.rs:207-215`
**Severity**: CRITICAL

```rust
*total_bytes += len;
if *total_bytes > MAX_TOTAL_STRING_DATA {
    return Err(...);
}
```

**Problem**: usize addition can overflow!
- total_bytes = usize::MAX - 1000
- len = 2000
- total_bytes += len → OVERFLOW → wraps to 999
- Check passes but we've wrapped around!

**Impact**: Can bypass 100MB limit via integer overflow

**Fix**: Use checked_add()

---

### 3. Iteration limit check happens AFTER work
**File**: `src/sentencepiece.rs:149-153`
**Severity**: HIGH

```rust
while let Some(bigram) = work_queue.pop() {
    iterations += 1;
    if iterations > max_iterations {
        return Err(...);
    }
```

**Problem**: We do work THEN check limit!
- max_iterations = 100K
- Iteration 100,001: pop bigram, do work, THEN error
- We did 100,001 iterations, not 100K

**Impact**: Off-by-one allows 1 extra iteration (minor)

**Fix**: Check BEFORE doing work

---

### 4. Recursion depth check happens AFTER recursive call
**File**: `src/sentencepiece.rs:341-348`
**Severity**: HIGH

```rust
fn resegment(..., depth: usize) {
    const MAX_RECURSION_DEPTH: usize = 1000;
    if depth > MAX_RECURSION_DEPTH {
        // fallback
        return;
    }
```

**Problem**: We check depth > 1000, but:
- Call resegment(..., 1000) → passes check
- That calls resegment(..., 1001) → fails, returns early
- But we're already 1001 stack frames deep!

**Impact**: Can reach 1001 depth instead of 1000

**Fix**: Check >= instead of >

---

### 5. Vec::with_capacity still allocates on empty input
**File**: `src/vocab.rs:80`
**Severity**: LOW

```rust
let mut token_to_id = HashMap::with_capacity(num_tokens);
```

**Problem**: If num_tokens = 0 (shouldn't happen due to validation), this works fine. But no validation that num_tokens isn't corrupted in memory.

**Impact**: Not a real bug (we validate num_tokens > 0)

**Status**: FALSE ALARM

---

### 6. bpe_fragment can return empty Vec on valid input
**File**: `src/bpe.rs:108-110`
**Severity**: MEDIUM

```rust
fn bpe_fragment(&self, text: &str, vocab: &Vocabulary) -> Result<Vec<TokenId>, crate::Error> {
    if text.is_empty() {
        return Ok(Vec::new());
    }
```

**Problem**: What if text = "\n\n\n" (all whitespace)?
- Pre-tokenization splits this into fragments
- Each fragment goes through bpe_fragment
- If fragment is empty string → return empty vec
- Result: input "\n\n\n" → 0 tokens!

**Impact**: Whitespace-only input produces no tokens (might be intentional?)

**Fix**: Verify this matches llama.cpp behavior

---

### 7. Error messages still leak token IDs
**Files**: `src/bpe.rs:298`, `src/sentencepiece.rs:263`
**Severity**: LOW

```rust
return Err(Error::InvalidToken(format!(
    "Token ID {} not found in vocabulary",
    token_id
)));
```

**Problem**: Attacker can probe vocabulary by trying token IDs and reading errors.

**Impact**: Information disclosure (acceptable for library, not network service)

**Status**: ACCEPTED RISK

---

### 8. MAX_DECODE_SIZE check uses byte_encoded_text.len()
**File**: `src/bpe.rs:313-318`
**Severity**: MEDIUM

```rust
const MAX_DECODE_SIZE: usize = 100 * 1024 * 1024; // 100MB
if byte_encoded_text.len() > MAX_DECODE_SIZE {
    return Err(...);
}
let decoded = crate::byte_encoder::decode_bytes(&byte_encoded_text);
```

**Problem**: 
- byte_encoded_text = 90MB (passes check)
- decode_bytes() expands it to 120MB (GPT-2 encoding can expand)
- Final decoded string is 120MB!

**Impact**: Can exceed 100MB limit after decoding

**Fix**: Check decoded.len() after decode_bytes(), or use lower limit

---

### 9. Merge validation only checks token exists, not bounds
**File**: `src/vocab.rs:103-115`
**Severity**: MEDIUM

```rust
if !token_to_id.contains_key(left) {
    return Err(Error::VocabularyError(...));
}
```

**Problem**: We check merge tokens exist in token_to_id HashMap, but:
- What if merged result token doesn't exist?
- What if left+right produces token longer than MAX_TOKEN_LENGTH?

**Impact**: Merge could produce invalid token

**Fix**: Also validate merged result exists in vocab

---

### 10. SentencePiece doesn't validate processed_text size
**File**: `src/sentencepiece.rs:92-97`
**Severity**: LOW

```rust
let processed_text = if !text.starts_with(' ') {
    format!("▁{}", text.replace(' ', "▁"))
} else {
    text.replace(' ', "▁")
};
```

**Problem**: 
- Input: 10MB of spaces
- processed_text: 10MB * 3 bytes (▁ is 3-byte UTF-8) = 30MB
- Could exceed MAX_INPUT_SIZE!

**Impact**: Input size check bypassed by space→▁ expansion

**Fix**: Check processed_text.len() after transformation

---

## HIGH PRIORITY ISSUES

### 11. No validation that vocab.tokens matches vocab.scores length
**File**: `src/vocab.rs:126`
**Severity**: MEDIUM

```rust
scores: metadata.scores.unwrap_or_else(|| vec![0.0; num_tokens]),
```

**Problem**: What if metadata.scores.len() != num_tokens?
- scores has 1000 entries
- tokens has 50K entries
- get_token_score(10000) → index out of bounds!

**Impact**: Panic on score lookup

**Fix**: Validate scores.len() == tokens.len() if scores provided

---

### 12. BPE pre-tokenization regex is never validated
**File**: `src/bpe.rs:77-89`
**Severity**: LOW

```rust
self.gpt2_regex.get_or_init(|| {
    regex::Regex::new(GPT2_PATTERN).expect("GPT-2 regex pattern is invalid")
});
```

**Problem**: Uses expect() which can panic!

**Impact**: Can panic on first use if regex is invalid

**Fix**: Validate at compile time or return Result

---

### 13. get_token_score has implicit bounds check via unwrap_or
**File**: `src/vocab.rs:154`
**Severity**: LOW

```rust
pub fn get_token_score(&self, id: TokenId) -> f32 {
    self.scores.get(id as usize).copied().unwrap_or(0.0)
}
```

**Problem**: If id >= scores.len(), returns 0.0 (default score).
- Is 0.0 the right default?
- What if vocab uses negative scores?

**Impact**: Silent fallback might be wrong

**Fix**: Document behavior or return Option<f32>

---

## MEDIUM PRIORITY ISSUES

### 14. Vocabulary merges are stored as Vec not validated length
**File**: `src/vocab.rs:140`
**Severity**: LOW

```rust
merges: metadata.merges.unwrap_or_default(),
```

**Problem**: We validate merge references but not merge count limit.
- Could have 10M merges
- Wastes memory

**Impact**: Memory usage unbounded for merges

**Fix**: Add MAX_MERGE_COUNT (e.g., 1M)

---

### 15. TokenType::from(i32) might be wrong for unknown values
**File**: `src/vocab.rs` (via TokenType enum)
**Severity**: LOW

**Problem**: What if GGUF has token_type = 999?

**Impact**: Depends on TokenType implementation

**Status**: NEEDS CODE REVIEW of TokenType enum

---

### 16. No concurrent access safety documented
**File**: `src/lib.rs`
**Severity**: MEDIUM

**Problem**: Is Tokenizer Send? Sync?
- Uses OnceLock (Send + Sync)
- Uses Box<dyn TokenizerImpl> (depends on impl)
- BPE has Regex (Sync but not Send?)

**Impact**: Unclear if safe for concurrent use

**Fix**: Add Send/Sync bounds or document limitations

---

## SUMMARY

**Critical Issues**: 2
- Integer overflow in total_bytes tracking
- Output token check happens after allocation

**High Priority**: 4
- Iteration limit off-by-one
- Recursion depth off-by-one
- Decode size check before expansion
- Score/token length mismatch

**Medium Priority**: 10
- Various validation gaps
- Edge case handling
- Documentation needs

**Total**: 16 issues (vs 20 R1, 11 R2)

**Trend**: Issues decreasing! Code getting more robust.

---

## Next Steps

1. Fix 2 CRITICAL issues (integer overflow, allocation order)
2. Fix 4 HIGH issues (off-by-one errors, validation gaps)
3. Address medium priority issues
4. Proceed to Round 4
