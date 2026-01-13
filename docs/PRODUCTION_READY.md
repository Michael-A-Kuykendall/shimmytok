# Production Readiness Checklist

## Status: BPE FIXED ✅ - Now completing production prep

### 1. Code Completion & Testing ⚠️ IN PROGRESS

#### BPE Algorithm
- [x] Fix merge validation bug (stale bigram text)
- [x] All 8 tests pass
- [x] Add edge case tests (8 comprehensive tests added)
- [x] Fix decode byte reversal bug
- [x] 16/16 tests passing

#### Test Coverage
- [x] Empty string handling
- [x] Unicode edge cases (emoji, CJK)
- [x] Long strings (200 repetitions tested)
- [x] Special characters (!@#$%^&*())
- [x] Newlines and tabs
- [x] Multiple spaces
- [x] Round-trip stability (encode→decode→encode)
- [ ] Error path testing
- [ ] Special tokens edge cases

#### Additional Tests Needed
- [ ] `tests/test_errors.rs` - Error handling paths
- [ ] SentencePiece edge case tests
- [ ] Multi-model validation (different GGUF files)

### 2. PPT Invariant System 🔒 TODO

#### Implementation
- [ ] Create `src/invariants.rs`
  ```rust
  pub fn assert_invariant(condition: bool, msg: &str, context: Option<&str>)
  pub fn clear_violation_log()
  pub fn get_violations() -> Vec<Violation>
  ```

#### Critical Invariants to Add
- [ ] Vocabulary: `token_id < vocab.n_tokens()`
- [ ] BPE: Merge rank validation before apply
- [ ] BPE: All output tokens valid IDs
- [ ] SentencePiece: Symbol length > 0
- [ ] Public API: UTF-8 validity on decode
- [ ] Round-trip: encode→decode preserves meaning

#### Contract Tests
- [ ] `tests/test_invariants.rs`
  - Test that invariants fire when violated
  - Test that violations logged correctly
  - Lock down critical paths

### 3. Code Cleanup 🧹 ✅ COMPLETE

#### Code Quality
- [x] Fix all clippy warnings (0 warnings on lib)
- [x] Run cargo fmt (all files formatted)
- [x] Verify no println! debug statements
- [x] All Default impls added where needed

#### Debug Cruft (gitignored, not in repo)
- All debug examples ignored via .gitignore
- All planning markdown ignored via .gitignore
- Clean repository with only production code

### 4. Documentation 📚 TODO

#### README.md Enhancement
- [ ] Quick start example (2 lines of code)
- [ ] Supported models table
- [ ] Feature comparison vs other crates
- [ ] Installation instructions
- [ ] Basic usage examples
- [ ] Link to docs.rs

#### CHANGELOG.md
- [ ] Create with v0.1.0 initial release
- [ ] Document: SentencePiece + BPE support
- [ ] Document: GGUF file loading
- [ ] Document: Pure Rust, no C++ deps

#### API Documentation
- [ ] All pub functions have doc comments
- [ ] All pub structs have doc comments
- [ ] Code examples in doc comments
- [ ] Error types documented
- [ ] Run `cargo doc` and verify quality

### 5. Public Release Prep 🚀 TODO

#### Cargo.toml Polish
- [ ] version = "0.1.0"
- [ ] description = "Pure Rust GGUF tokenizer (SentencePiece + BPE)"
- [ ] license = "MIT" (or Apache-2.0, or MIT/Apache-2.0)
- [ ] repository = "https://github.com/USER/shimmytok"
- [ ] keywords = ["tokenizer", "gguf", "llm", "sentencepiece", "bpe"]
- [ ] categories = ["text-processing", "parsing"]
- [ ] readme = "README.md"
- [ ] exclude debug files from package

#### Legal
- [ ] Add LICENSE file
- [ ] Add copyright headers if needed
- [ ] Verify all dependencies have compatible licenses

#### Quality Gates
- [ ] `cargo test` - all pass
- [ ] `cargo clippy` - zero warnings
- [ ] `cargo fmt --check` - formatted
- [ ] `cargo doc` - builds cleanly
- [ ] `cargo package` - validates
- [ ] Manual test on fresh checkout

#### Publishing
- [ ] `cargo publish --dry-run`
- [ ] Review generated package
- [ ] `cargo publish`
- [ ] Verify on crates.io
- [ ] GitHub release tag v0.1.0

---

## Test Coverage Requirements

**Minimum before release:**
- 100% of public API exercised
- Edge cases covered (empty, unicode, long)
- Error paths tested
- Round-trip stability verified
- Multiple models validated (GPT-2, LLaMA minimum)

**Current Status:** 16 tests passing, ~85% coverage estimate
**Target:** 18-20 tests, 95%+ coverage

---

## File Counts

**Current:**
- 21+ debug examples to delete
- 13+ planning markdown files to delete
- ~50 files total

**Target:**
- 0-2 production examples
- 2-3 markdown files (README, CHANGELOG, maybe CONTRIBUTING)
- ~15 files total (lean & clean)
