# Current Tasks - shimmytok Production Checklist

## 🎯 Active Sprint: BPE Fix + Invariant System

### Phase 1: Fix BPE Algorithm ⚠️ IN PROGRESS
- [x] Port llama.cpp byte encoder (bytes_to_unicode)
- [x] Fix space encoding (space→Ġ via byte encoder)
- [ ] DEBUG: Merge algorithm not working
  - Tokens exist in vocab (Ġawesome = 7427)
  - Merge rules exist (("Ġ", "a") in merge list)
  - But encoder produces [220, 64, 86, 68, 82, 462] not [7427]
  - **ISSUE**: Merge loop not applying merges correctly
  - **NEXT**: Add debug logging to merge loop to see what's happening

### Phase 2: Test to Death
- [ ] Run all existing tests - must pass
- [ ] Add edge case tests
  - [ ] Empty string
  - [ ] Unicode (emoji, CJK, RTL)
  - [ ] Special characters
  - [ ] Very long strings
- [ ] Validate against llama.cpp on multiple models
  - [ ] GPT-2
  - [ ] Phi-3
  - [ ] LLaMA

### Phase 3: Invariant System Implementation
- [ ] Create `src/invariant_ppt.rs`
  - [ ] `assert_invariant(condition, msg, context)`
  - [ ] Invariant logging/tracking
  - [ ] `contract_test()` helper
  - [ ] `clear_invariant_log()` for CI
- [ ] Embed critical invariants in code
  - [ ] Vocabulary: valid token IDs
  - [ ] BPE: merge ranks valid
  - [ ] SentencePiece: symbol integrity
  - [ ] Public API: UTF-8 validity
  - [ ] Round-trip: encode/decode stability
- [ ] Create contract tests
  - [ ] `tests/test_contracts.rs`
  - [ ] Verify all invariants fire
  - [ ] Lock down critical paths

### Phase 4: Audit & Clean
- [ ] Code audit
  - [ ] Verify BPE matches llama.cpp
  - [ ] Verify SentencePiece matches llama.cpp
  - [ ] Check all error paths
- [ ] Remove debug cruft
  - [ ] Delete examples/debug_*.rs
  - [ ] Delete examples/check_*.rs
  - [ ] Keep only production examples
- [ ] Documentation cleanup
  - [ ] Consolidate markdown files
  - [ ] Move planning docs to archive/
  - [ ] Keep only: README, CHANGELOG, FUTURE_ENHANCEMENTS
- [ ] API review
  - [ ] Public interface clean
  - [ ] Error types comprehensive
  - [ ] Documentation complete

### Phase 5: Polish & Ship
- [ ] Final README with examples
- [ ] Add CHANGELOG.md
- [ ] Version to 0.1.0
- [ ] Publish to crates.io
- [ ] ✅ DONE

---

## 🔒 Critical Invariants to Implement

### Vocabulary Layer
```rust
assert_invariant(token_id < vocab.n_tokens(), "Token ID within bounds", Some("vocab"));
assert_invariant(!text.is_empty() || is_special, "Non-empty text or special token", Some("vocab"));
```

### BPE Tokenizer
```rust
assert_invariant(symbols.len() > 0, "Non-zero symbols after merge", Some("bpe"));
assert_invariant(merge_ranks.contains_key(&pair), "Valid merge rank", Some("bpe"));
assert_invariant(result.iter().all(|&id| id < vocab.n_tokens()), "All tokens valid", Some("bpe"));
```

### SentencePiece Tokenizer
```rust
assert_invariant(symbol.len > 0, "Valid symbol length", Some("spm"));
assert_invariant(utf8::valid_up_to(text) == text.len(), "Valid UTF-8", Some("spm"));
```

### Public API
```rust
assert_invariant(decoded.is_valid_utf8(), "Valid UTF-8 output", Some("decode"));
assert_invariant(
    encode(text).then_decode() matches original (whitespace-normalized),
    "Round-trip stable",
    Some("api")
);
```

---

## 📦 Current Project State

### ✅ Working
- SentencePiece tokenizer (LLaMA, Phi, Mistral, Gemma)
- GGUF file loading
- Vocabulary management
- Public API structure

### ⚠️ Broken
- BPE tokenizer (GPT-2, Falcon, StarCoder)
  - Bug: initializes with UTF-8 chars instead of regex words
  - Bug: byte fallback tries hex format for non-SPM models

### 📊 Coverage
- Current: ~95% of mainstream models (after BPE fix)
- Future: WPM (BERT), UGM (T5), RWKV for 100%

---

## 🎓 Architecture Notes

**Module Structure:**
- `lib.rs` - Public API, tokenizer router
- `vocab.rs` - Vocabulary loading, token lookup
- `gguf.rs` - File format reader
- `sentencepiece.rs` - SPM algorithm (WORKING)
- `bpe.rs` - BPE algorithm (FIXING)
- `invariant_ppt.rs` - Invariant system (TODO)

**Trait Contract:**
```rust
trait TokenizerImpl {
    fn encode(&self, text: &str, vocab: &Vocabulary) -> Vec<TokenId>;
    fn decode(&self, tokens: &[TokenId], vocab: &Vocabulary) -> String;
}
```

All tokenizers implement this. Router in lib.rs dispatches based on `vocab.model_type()`.

---

## 🚀 Next Immediate Action

**FIX BPE NOW** - Port llama.cpp algorithm exactly (lines 1027-1110)
