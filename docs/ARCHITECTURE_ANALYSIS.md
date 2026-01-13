# Architecture Analysis: BPE Algorithm Change Impact

## Current Architecture

### Module Structure
```
lib.rs (Public API)
├── Tokenizer (main interface)
│   ├── from_gguf_file() - entry point
│   ├── encode() - adds BOS/EOS, delegates to impl
│   ├── decode() - filters special tokens, delegates to impl
│   └── tokenizer_impl: Box<dyn TokenizerImpl>
│
├── vocab.rs (Vocabulary)
│   ├── from_gguf_file() - loads GGUF metadata
│   ├── get_token_id(text) -> Option<TokenId>
│   ├── get_token_text(id) -> Option<&str>
│   ├── get_token_score(id) -> f32
│   ├── byte_to_token(byte) -> TokenId
│   ├── get_merges() -> &[(String, String)]
│   └── model_type() -> &str
│
├── sentencepiece.rs (SentencePieceTokenizer)
│   └── impl TokenizerImpl
│       ├── encode(text, vocab) -> Vec<TokenId>
│       └── decode(tokens, vocab) -> String
│
├── bpe.rs (BPETokenizer) ⚠️ TARGET OF CHANGE
│   └── impl TokenizerImpl
│       ├── encode(text, vocab) -> Vec<TokenId>
│       └── decode(tokens, vocab) -> String
│
└── gguf.rs (File reader)
    └── load_metadata() -> GGUFMetadata
```

### TokenizerImpl Trait Contract
```rust
trait TokenizerImpl {
    fn encode(&self, text: &str, vocab: &Vocabulary) -> Vec<TokenId>;
    fn decode(&self, tokens: &[TokenId], vocab: &Vocabulary) -> String;
}
```

**CRITICAL**: Both methods take `&Vocabulary` as parameter. They DON'T modify vocab, only read from it.

## Current BPE Implementation (src/bpe.rs)

### Algorithm Flow
1. **Pre-tokenization**: Uses regex to split text into fragments
2. **Initial symbols**: Creates UTF-8 character symbols for each fragment
3. **Merge loop**: Uses priority queue with merge ranks from vocab
4. **Byte fallback**: Uses `vocab.byte_to_token(byte)` for unknown tokens

### Key Dependencies
- `Vocabulary::get_token_id(text)` - lookup merged text
- `Vocabulary::get_merges()` - get (token1, token2) -> rank pairs
- `Vocabulary::byte_to_token(byte)` - fallback for unknowns
- `Vocabulary::get_token_text(id)` - for decode

## Proposed llama.cpp BPE Port

### Algorithm Changes
1. **Pre-tokenization**: Same (regex split)
2. **Initial symbols**: ⚠️ DIFFERENT - whole words from regex, not characters
3. **Merge loop**: Same concept, different initialization
4. **Byte fallback**: ⚠️ DIFFERENT - lookup bytes as strings in vocab directly

### Modified Dependencies
- `Vocabulary::get_token_id(text)` - ✅ SAME
- `Vocabulary::get_merges()` - ✅ SAME  
- `Vocabulary::byte_to_token(byte)` - ⚠️ USAGE CHANGES (see below)
- `Vocabulary::get_token_text(id)` - ✅ SAME

## Impact Analysis

### 1. Vocabulary Interface (vocab.rs)
**Status**: ✅ NO CHANGES NEEDED

Current interface supports both approaches:
- `get_token_id()` works for any text (characters OR words)
- `get_merges()` returns same data structure
- `byte_to_token()` exists but behavior may need adjustment

### 2. Byte Fallback Issue
**Current Implementation** (vocab.rs:112-123):
```rust
pub fn byte_to_token(&self, byte: u8) -> TokenId {
    // Try hex format <0xXX> first (SPM style)
    let hex_str = format!("<0x{:02X}>", byte);
    if let Some(id) = self.token_to_id.get(&hex_str) {
        return *id;
    }
    
    // Fall back to raw byte as string
    let byte_str = String::from_utf8_lossy(&[byte]).to_string();
    self.token_to_id.get(&byte_str).copied()
        .unwrap_or(self.unk_token_id)
}
```

**llama.cpp Approach** (line 1107):
```cpp
for (auto j = str.begin(); j != str.end(); ++j) {
    std::string byte_str(1, *j);  // Single byte as string
    auto token_multibyte = vocab.text_to_token(byte_str);
    if (token_multibyte != LLAMA_TOKEN_NULL) {
        output.push_back(token_multibyte);
    }
}
```

**Analysis**:
- llama.cpp does: lookup each byte as a 1-char string in vocab
- Current code does: same thing (second fallback path)
- ✅ **COMPATIBLE** - our `byte_to_token()` already does this

**Issue**: The hex format lookup happens FIRST but shouldn't for BPE.

**Solution**: 
- Option A: Add parameter `use_hex_format: bool` to `byte_to_token()`
- Option B: Create `byte_to_token_bpe()` method that skips hex lookup
- Option C: Check vocab.model_type() inside `byte_to_token()` and skip hex for BPE

### 3. SentencePiece Tokenizer (sentencepiece.rs)
**Status**: ✅ NO IMPACT

SentencePiece:
- Uses different initialization (UTF-8 chars with ▁ prefix)
- Uses resegment algorithm (not affected by BPE changes)
- Calls `vocab.byte_to_token()` only for fallback
- Won't be touched during BPE refactor

### 4. Public API (lib.rs)
**Status**: ✅ NO CHANGES NEEDED

```rust
impl Tokenizer {
    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<TokenId>, Error>
    pub fn decode(&self, tokens: &[TokenId], skip_special: bool) -> Result<String, Error>
}
```

- Same method signatures
- Same behavior from user perspective
- Only internal algorithm changes

### 5. Test Suite
**Files**: tests/test_bpe.rs, tests/test_comprehensive.rs

**Expected Impact**: ✅ TESTS SHOULD NOW PASS

Current tests:
- Round-trip encode/decode
- Comparison with llama.cpp output
- Currently FAILING because of algorithm bug

After fix:
- Should produce correct token IDs
- Should match llama.cpp exactly
- Round-trip should work

### 6. GGUF Loader (gguf.rs)
**Status**: ✅ NO IMPACT

- Reads file format
- Parses metadata
- Doesn't care about algorithm

## Compatibility Matrix

| Component | Current BPE | llama.cpp BPE | Compatible? |
|-----------|-------------|---------------|-------------|
| Vocabulary API | read-only | read-only | ✅ YES |
| TokenizerImpl trait | encode/decode | encode/decode | ✅ YES |
| Special token handling | lib.rs | lib.rs | ✅ YES |
| Regex pre-tokenize | yes | yes | ✅ YES |
| Symbol initialization | UTF-8 chars | regex words | ⚠️ DIFFERENT (BPE only) |
| Merge algorithm | priority queue | priority queue | ✅ SAME |
| Byte fallback | byte_to_token() | lookup string | ⚠️ NEEDS FIX |
| Decode | get_token_text() | get_token_text() | ✅ YES |

## Risk Assessment

### Low Risk ✅
- Public API unchanged
- Vocabulary interface unchanged  
- SentencePiece unaffected
- GGUF parsing unaffected
- Test expectations aligned

### Medium Risk ⚠️
- `byte_to_token()` behavior difference between SPM and BPE
- Need to verify GPT-2 vocab has byte tokens as strings (not hex)

### High Risk ❌
- NONE IDENTIFIED

## Recommendation

### ✅ SAFE TO PROCEED

The change is **architecturally sound** because:

1. **Isolation**: Only `bpe.rs` changes, other modules unchanged
2. **Interface Stability**: No trait or public API changes
3. **Backward Compatible**: SentencePiece still works
4. **Test Coverage**: Existing tests will validate correctness

### Required Changes

1. **bpe.rs** (256 lines):
   - Change initial symbol creation (whole words not chars)
   - Update byte fallback to use direct vocab lookup
   - Port llama.cpp merge algorithm exactly

2. **vocab.rs** (1 method):
   - Fix `byte_to_token()` to check model type OR
   - Create separate `byte_to_token_bpe()` method OR
   - Let BPE call `get_token_id()` directly instead

### Verification Steps

1. Run existing test suite - should now PASS
2. Compare with llama.cpp output - should MATCH
3. Test round-trip encoding - should WORK
4. Verify SentencePiece still works - should be UNCHANGED

## Decision: PROCEED ✅

The architectural analysis shows **no blocking issues**. The change is:
- Isolated to one module
- Interface-compatible
- Low-risk
- Will fix current bugs
- Aligns with llama.cpp reference implementation
