# Hostile Audit Round 2

## Audit Date: 2025-10-21

### Context
Round 1 fixed 7 CRITICAL issues by adding Result-based error handling and input validation.
This round audits the NEW error handling code for flaws.

### Methodology
Focus on:
- Error handling paths (are they correct?)
- Edge cases in new validation logic
- Resource exhaustion still possible?
- Error messages leak sensitive data?
- New code introduce bugs?

---

## CRITICAL ISSUES

### 1. Input size validation is INCOMPLETE
**Files**: `src/bpe.rs:48`, `src/sentencepiece.rs:79`
**Severity**: CRITICAL

```rust
const MAX_INPUT_SIZE: usize = 10 * 1024 * 1024; // 10MB
if text.len() > MAX_INPUT_SIZE {
    return Err(Error::TokenizationFailed(...));
}
```

**Problem**: This only checks UTF-8 byte length. What about:
- Text with 10MB of ASCII = 10M chars → OK
- Text with 10MB of 4-byte UTF-8 = 2.5M chars → OK
- Text with pathological BPE patterns = could produce 10M tokens → NOT CHECKED

The token output size is unbounded!

**Impact**: 
- Encode "aaaaaaa..." (10MB) → Could produce 10M tokens → 40MB Vec<u32>
- Memory exhaustion bypass

**Fix**: Add MAX_OUTPUT_TOKENS limit (e.g., 1M tokens)

---

### 2. Iteration limit in SentencePiece is TOO HIGH
**File**: `src/sentencepiece.rs:147`
**Severity**: HIGH

```rust
let max_iterations = 10 * symbols.len().max(1);
```

**Problem**: For 10MB input, this could be:
- 10MB text → ~10M symbols initially
- max_iterations = 100M iterations!
- Each iteration does HashMap lookups, heap operations

**Impact**: DoS - 100M iterations takes seconds/minutes

**Fix**: Cap at reasonable value like 100K iterations regardless of input size

---

### 3. GGUF string validation allows 1MB EACH
**File**: `src/gguf.rs:169`
**Severity**: HIGH

```rust
const MAX_STRING_SIZE: usize = 1024 * 1024; // 1MB
```

**Problem**: GGUF files have multiple strings:
- model.name
- tokenizer.ggml.model
- tokenizer.ggml.tokens (array of strings!)
- 1000 tokens * 1MB each = 1GB allocation!

**Impact**: Still vulnerable to OOM via many 1MB strings

**Fix**: Add total string data limit (e.g., 100MB total)

---

### 4. Vocabulary validation doesn't check token_id bounds
**File**: `src/vocab.rs:56-96`
**Severity**: CRITICAL

```rust
for (i, token) in tokens.iter().enumerate() {
    let token_id = i as u32;
    // ... validate token text ...
}
```

**Problem**: What if GGUF has:
- 50K tokens → token_ids 0-49999 ✓
- But merge pairs reference token_id 50000? 
- Or vocab.id_to_text HashMap has gaps?

We never validate that ALL referenced token_ids are contiguous!

**Impact**: BPE merge can panic on out-of-bounds access

**Fix**: Validate merge pairs reference valid token_ids

---

### 5. Error::InvalidToken leaks vocabulary contents
**File**: `src/lib.rs:23`
**Severity**: MEDIUM

```rust
InvalidToken(String),
```

Used in:
```rust
return Err(Error::InvalidToken(format!(
    "Token ID {} not found in vocabulary",
    token_id
)));
```

**Problem**: If attacker controls token_ids, error messages leak which IDs exist.
Could use this to map vocabulary structure.

**Impact**: Information disclosure (minor but enumeration attack)

**Fix**: Don't include token_id in error, or rate-limit errors

---

### 6. BPE pre_tokenize regex can catastrophic backtrack
**File**: `src/bpe.rs:27`
**Severity**: HIGH

```rust
let re = Regex::new(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
).expect("Invalid regex pattern");
```

**Problem**: Alternation with `?` quantifiers can backtrack on pathological input:
- Input: "  \u{200B}  \u{200B}  \u{200B}" (many zero-width spaces)
- Regex tries all alternations for each position
- Exponential time complexity

**Impact**: ReDoS (Regex DoS) - freeze on specially crafted input

**Fix**: Use atomic groups or possessive quantifiers, or limit regex execution time

---

### 7. Decode doesn't validate output size
**Files**: `src/bpe.rs:292`, `src/sentencepiece.rs:263`
**Severity**: MEDIUM

```rust
pub fn decode(&self, tokens: &[TokenId], vocab: &Vocabulary) -> Result<String, Error> {
    for &token_id in tokens {
        if vocab.get_token_text(token_id).is_none() { ... }
    }
    // ... build string ...
}
```

**Problem**: Input = 1M valid token_ids, each maps to 1KB string → 1GB output String

**Impact**: Memory exhaustion on decode path

**Fix**: Add MAX_DECODE_SIZE limit (e.g., 100MB output)

---

### 8. Vocabulary HashMap has no capacity hint
**File**: `src/vocab.rs:93`
**Severity**: LOW

```rust
let mut id_to_text = HashMap::new();
for (i, token) in tokens.iter().enumerate() {
    id_to_text.insert(i as u32, token.clone());
}
```

**Problem**: HashMap resizes multiple times during insert (50K tokens → ~4 resizes)

**Impact**: Performance - unnecessary allocations

**Fix**: `HashMap::with_capacity(tokens.len())`

---

### 9. SentencePiece resegment is recursive
**File**: `src/sentencepiece.rs:292-339`
**Severity**: MEDIUM

```rust
fn resegment(
    text: &str,
    // ...
) {
    if text.is_empty() {
        return;
    }
    // ... 
    // No explicit recursion seen but called in loop
}
```

**Problem**: Actually not recursive after Round 1 fix, but let me verify...

[Checking if there's hidden recursion in resegment calls]

**Status**: NEEDS VERIFICATION

---

### 10. TokenId is u32 but vocab uses usize indexes
**Files**: `src/lib.rs:51`, `src/vocab.rs:56`
**Severity**: LOW

```rust
pub type TokenId = u32;
```

But:
```rust
let token_id = i as u32; // i is usize
```

**Problem**: On 64-bit systems, usize can exceed u32::MAX. If vocab has > 4B tokens, this truncates.

**Impact**: 
- We limit vocab to 1M (MAX_VOCAB_SIZE) so this can't happen
- But inconsistent types are confusing

**Fix**: Document why u32 is safe, or use stronger type safety

---

## HIGH PRIORITY ISSUES

### 11. BPE byte_encoder is pre-allocated globally
**File**: `src/bpe.rs:12-13`
**Severity**: MEDIUM

```rust
lazy_static! {
    static ref BYTE_ENCODER: HashMap<u8, char> = create_byte_encoder();
}
```

**Problem**: Uses lazy_static which adds dependency and initialization overhead

**Impact**: Not a bug but could use const fn instead (Rust 1.57+)

**Fix**: Consider const-based approach for zero-cost initialization

---

### 12. GGUF read_string allocates full buffer before validation
**File**: `src/gguf.rs:169-177`
**Severity**: MEDIUM

```rust
const MAX_STRING_SIZE: usize = 1024 * 1024; // 1MB
if len > MAX_STRING_SIZE {
    return Err(...);
}
let mut buffer = vec![0u8; len as usize];
reader.read_exact(&mut buffer)?;
```

**Problem**: This is correct! Validates BEFORE allocation.

**Status**: FALSE ALARM - This is actually good code

---

## MEDIUM PRIORITY ISSUES

### 13. Error types don't implement Send + Sync
**File**: `src/lib.rs:18-28`
**Severity**: MEDIUM

```rust
#[derive(Debug, Clone)]
pub enum Error {
    // ...
}
```

**Problem**: For async/concurrent use, errors should be Send + Sync

**Impact**: Can't use tokenizer in async contexts or across threads

**Fix**: Ensure all error variants are Send + Sync (currently they are, but not explicit)

---

### 14. Vocabulary::get_token_text returns Option<&str> requiring linear scan?
**File**: `src/vocab.rs:122-125`
**Severity**: LOW

```rust
pub fn get_token_text(&self, token_id: TokenId) -> Option<&str> {
    self.id_to_text.get(&token_id).map(|s| s.as_str())
}
```

**Problem**: Wait, this uses HashMap.get() which is O(1). The comment is misleading.

**Status**: FALSE ALARM - Performance is fine

---

### 15. Tests don't verify error cases
**Files**: `tests/*.rs`
**Severity**: MEDIUM

**Problem**: All tests check happy path. None verify:
- Encoding 11MB input → TokenizationFailed
- Decoding invalid token_id → InvalidToken
- Loading corrupt GGUF → InvalidMetadata
- etc.

**Impact**: Error handling is untested

**Fix**: Add negative test cases for each error path

---

## SUMMARY

**Critical Issues Found**: 4
- Input size validation incomplete (output unbounded)
- Iteration limit too high
- GGUF total string size unbounded
- Vocabulary doesn't validate token_id bounds in merges

**High Priority**: 2
- Regex backtracking DoS
- Decode output size unbounded

**Medium Priority**: 5
- Error message information disclosure
- Missing capacity hints
- Resegment verification needed
- Type inconsistency (usize→u32)
- No negative tests

**Total Issues**: 11 (vs 20 in Round 1)

**Progress**: 45% reduction in issues! Code is getting more robust.

---

## Next Steps

1. Fix 4 CRITICAL issues
2. Add output size limits to encode/decode
3. Cap iteration limits reasonably
4. Validate merge pair token_ids
5. Add GGUF total allocation limit
6. Then proceed to Round 3
