# Hostile Audit Round 4

## Audit Date: 2025-10-22

### Context
- Round 1: Fixed 20 issues
- Round 2: Fixed 11 issues
- Round 3: Fixed 16 issues
- Total: 47 issues fixed across 3 rounds

This round focuses on interaction between fixes and remaining edge cases.

### Methodology
- Test interactions between validation layers
- Look for logic errors in complex paths
- Verify error messages are consistent
- Check for resource leaks on error paths
- Look for remaining panics

---

## CRITICAL ISSUES

### 1. checked_add error doesn't restore state
**File**: `src/gguf.rs:211-214`
**Severity**: MEDIUM

```rust
*total_bytes = total_bytes.checked_add(len)
    .ok_or_else(|| Error::InvalidMetadata(
        "Total string data overflow".to_string()
    ))?;
```

**Problem**: If checked_add fails, we return error, but what about the string we just read?
- We've already called `reader.read_exact(&mut buf)`
- Reader position has advanced
- Can't retry or recover

**Impact**: Not a bug - error is fatal anyway

**Status**: FALSE ALARM - error handling is correct

---

### 2. Multiple MAX_INPUT_SIZE constants
**Files**: `src/bpe.rs:262`, `src/sentencepiece.rs:79`
**Severity**: LOW

```rust
const MAX_INPUT_SIZE: usize = 10 * 1024 * 1024; // 10MB
```

**Problem**: Defined in both files! If we change one, need to change both.

**Impact**: Code duplication, maintenance hazard

**Fix**: Define once in lib.rs as public constant

---

### 3. BPE fragment allocation still unbounded per fragment
**File**: `src/bpe.rs:268-279`
**Severity**: LOW

```rust
for fragment in fragments {
    let tokens = self.bpe_fragment(&fragment, vocab)?;
    if result.len() + tokens.len() > MAX_OUTPUT_TOKENS {
        return Err(...);
    }
    result.extend(tokens);
}
```

**Problem**: Single fragment could produce 500K tokens before we check.
- We allocate Vec in bpe_fragment
- Return it
- Then check size
- Still wasting allocation

**Impact**: Minor - can't avoid allocation before checking

**Status**: ACCEPTED LIMITATION

---

### 4. SentencePiece iteration limit can trigger on valid input
**File**: `src/sentencepiece.rs:147`
**Severity**: MEDIUM

```rust
let max_iterations = (10 * symbols.len()).min(100_000);
```

**Problem**: For 15K symbols:
- max_iterations = 100K (capped)
- What if valid tokenization needs 101K iterations?
- We'd error on legitimate input!

**Impact**: Potential false positives (but rare)

**Fix**: Ensure 100K is enough for real-world use

**Status**: NEEDS TESTING with large inputs

---

### 5. Recursion depth limit might be too low
**File**: `src/sentencepiece.rs:344`
**Severity**: LOW

```rust
const MAX_RECURSION_DEPTH: usize = 1000;
if depth >= MAX_RECURSION_DEPTH {
```

**Problem**: Stack depth of 1000 seems high, but:
- Each frame is small
- Rust default stack is 2MB
- 1000 frames * 2KB each = 2MB!

**Impact**: Could actually hit stack overflow before hitting limit

**Fix**: Reduce to 500 or measure actual frame size

---

### 6. Error handling doesn't free intermediate allocations
**File**: `src/bpe.rs:276-279`
**Severity**: LOW

```rust
if result.len() + tokens.len() > MAX_OUTPUT_TOKENS {
    return Err(...);
}
result.extend(tokens);
```

**Problem**: On error, `result` Vec is dropped, but we've allocated:
- `result` (could be 500K tokens = 2MB)
- `tokens` (could be 500K tokens = 2MB)
- Both dropped on error = 4MB wasted

**Impact**: Normal Rust behavior, not a bug

**Status**: FALSE ALARM

---

### 7. UTF-8 validation in decode is incomplete
**File**: `src/bpe.rs:322-327`
**Severity**: LOW

```rust
if !decoded.is_empty() && decoded.as_bytes().iter().any(|&b| b == 0xEF && decoded.contains('�')) {
    return Err(...);
}
```

**Problem**: This check is convoluted and wrong!
- Checks if byte 0xEF exists AND string contains �
- 0xEF is valid UTF-8 (start of 3-byte sequence)
- This could false positive on valid text!

**Impact**: Could reject valid Japanese/Chinese text

**Fix**: Remove this check - decode_bytes already handles it

---

### 8. No limit on individual fragment size
**File**: `src/bpe.rs:258-262`
**Severity**: MEDIUM

```rust
const MAX_INPUT_SIZE: usize = 10 * 1024 * 1024; // 10MB
if text.len() > MAX_INPUT_SIZE {
    return Err(...);
}
let text_encoded = crate::byte_encoder::encode_bytes(text);
let fragments = self.pre_tokenize(&text_encoded, vocab)?;
```

**Problem**: 
- Input is 10MB → passes
- encode_bytes() could expand (GPT-2 encoding)
- Single fragment could be entire 10MB
- bpe_fragment() processes it with no size check

**Impact**: Can process very large fragments

**Fix**: Add fragment size limit in bpe_fragment

---

### 9. Vocabulary get_token_score silently returns 0.0
**File**: `src/vocab.rs:169`
**Severity**: LOW

```rust
pub fn get_token_score(&self, id: TokenId) -> f32 {
    self.scores.get(id as usize).copied().unwrap_or(0.0)
}
```

**Problem**: What if token exists but has no score?
- We validated scores.len() == tokens.len()
- So if id < tokens.len(), score exists
- But we use unwrap_or(0.0) anyway

**Impact**: Dead code - unwrap_or never triggers

**Fix**: Remove unwrap_or or add comment

---

### 10. get_token_type also has unreachable unwrap_or
**File**: `src/vocab.rs:172-176`
**Severity**: LOW

```rust
pub fn get_token_type(&self, id: TokenId) -> TokenType {
    self.token_types
        .get(id as usize)
        .copied()
        .unwrap_or(TokenType::Undefined)
}
```

**Problem**: Same as #9 - we validated token_types.len() == tokens.len()

**Impact**: Dead code

**Fix**: Remove unwrap_or or panic on invalid ID

---

## HIGH PRIORITY ISSUES

### 11. TokenId cast from usize can panic on 32-bit systems
**File**: `src/vocab.rs:80`
**Severity**: LOW

```rust
for (i, s) in metadata.tokens.iter().enumerate() {
    token_to_id.insert(s.clone(), i as u32);
}
```

**Problem**: On 32-bit systems:
- usize is 32-bit
- MAX_VOCAB_SIZE = 1M < u32::MAX
- So this is safe

**Impact**: None - already protected by MAX_VOCAB_SIZE

**Status**: FALSE ALARM

---

### 12. GGUF read_u64 cast to usize can truncate
**File**: `src/gguf.rs:200`
**Severity**: MEDIUM

```rust
let len = read_u64(reader)? as usize;
```

**Problem**: On 32-bit systems:
- read_u64 returns u64 (0 to 2^64)
- Cast to usize (0 to 2^32 on 32-bit)
- Could truncate!

**Impact**: Can bypass length checks on 32-bit systems

**Fix**: Check len fits in usize before cast

---

### 13. Empty string fragments produce no tokens
**File**: `src/bpe.rs:108-110`
**Severity**: LOW

```rust
if text.is_empty() {
    return Ok(Vec::new());
}
```

**Problem**: Regex could produce empty fragments
- "  hello  " → fragments: ["", "hello", ""]
- Each "" → 0 tokens
- Is this correct?

**Impact**: Depends on regex behavior

**Status**: NEEDS VERIFICATION

---

## MEDIUM PRIORITY ISSUES

### 14. Error types don't distinguish temporary vs permanent failures
**File**: `src/lib.rs:18-28`
**Severity**: LOW

**Problem**: All errors are permanent - can't retry
- TokenizationFailed - is it data or algorithm?
- InvalidMetadata - corrupted file?
- No way to know if retry would help

**Impact**: Poor error handling by users

**Fix**: Add error categories

---

### 15. No way to cancel long-running tokenization
**File**: All encode functions
**Severity**: MEDIUM

**Problem**: If tokenization takes 10 seconds:
- No way to cancel
- No progress callback
- Blocks thread completely

**Impact**: Can't use in interactive contexts

**Fix**: Accept cancellation token or timeout

---

### 16. Vocabulary is not Send/Sync verified
**File**: `src/vocab.rs`
**Severity**: LOW

**Problem**: Contains:
- HashMap (Send + Sync)
- Vec (Send + Sync)  
- String (Send + Sync)
- Should be Send + Sync but not explicitly verified

**Impact**: Might not work in async contexts

**Fix**: Add static assertions

---

## SUMMARY

**Critical Issues**: 0
**High Priority**: 1 (u64→usize truncation on 32-bit)
**Medium Priority**: 15

**Total**: 16 issues (same as Round 3!)

**Observations**:
- Most issues are code quality, not correctness
- Many are dead code or overly defensive
- Real bugs are rare now
- UTF-8 validation logic is wrong

---

## Next Steps

1. Fix UTF-8 validation (Issue #7) - actually broken
2. Fix u64→usize truncation (Issue #12) - real security issue
3. Remove dead code (unwrap_or on validated data)
4. Consolidate constants
5. Proceed to Round 5 (final)
