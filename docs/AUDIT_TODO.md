# Shimmytok v0.6.0 - Deep Audit & Hardening TODO

## Current Status (Baseline v0.6.0)
- ✅ BPE: 6 models validated at 100% (GPT-2, Qwen2, StarCoder, DeepSeek x2, Phi-2)
- ✅ SentencePiece: Fully validated with resegment algorithm
- ✅ ~95% coverage of popular model families (41 pre-tokenizer types from llama.cpp)
- ✅ All unit tests passing (27 tests)
- ⚠️ 22 dead code warnings (old pattern constants)
- ⚠️ Minimal unit tests for multi-pattern splitting logic
- ⚠️ No benchmarks

## Phase 1: Code Quality & Best Practices (8 Fibonacci points)

### 1.1 Clean Up Dead Code (2 points)
- [ ] Remove all unused pattern constants (GPT2_PATTERN, LLAMA3_PATTERN, etc.)
- [ ] Verify no other dead code with `cargo clippy`
- [ ] Run `cargo fmt` for consistent formatting

### 1.2 Rust Best Practices Audit (3 points)
- [ ] Run `cargo clippy -- -W clippy::all -W clippy::pedantic`
- [ ] Fix all clippy warnings
- [ ] Review error handling - ensure all `unwrap()` are justified or replaced
- [ ] Check for unnecessary clones/allocations
- [ ] Review visibility modifiers (pub vs pub(crate))

### 1.3 Documentation (3 points)
- [ ] Add module-level docs for bpe.rs, vocab.rs, sentencepiece.rs
- [ ] Document the offset-based multi-pattern algorithm
- [ ] Add inline comments for complex logic (especially BPE merge algorithm)
- [ ] Verify all public APIs have doc comments
- [ ] Add examples to tricky functions

## Phase 2: Test Coverage Expansion (13 Fibonacci points)

### 2.1 Unit Tests for Core Logic (5 points)
- [ ] Test offset-based splitting with 2, 3, 5 patterns
- [ ] Test gap preservation in multi-pattern scenarios
- [ ] Test pattern matching edge cases (empty strings, no matches, all matches)
- [ ] Test DEFAULT fallback explicitly
- [ ] Test byte encoder edge cases

### 2.2 Multi-Pattern Edge Cases (3 points)
- [ ] Test patterns that produce no matches
- [ ] Test patterns with overlapping match ranges
- [ ] Test very long input with multi-pattern splitting
- [ ] Test unicode edge cases in multi-pattern mode

### 2.3 BPE Merge Algorithm Tests (3 points)
- [ ] Test merge priority with complex ranking scenarios
- [ ] Test symbol linking correctness
- [ ] Test work queue behavior
- [ ] Test empty/single-character inputs

### 2.4 Integration Tests (2 points)
- [ ] Add validation tests for more model types (aim for 10+ models)
- [ ] Test round-trip encode/decode for all supported types
- [ ] Test batch processing with mixed model types

## Phase 3: Performance & Robustness (8 Fibonacci points)

### 3.1 Benchmarks (3 points)
- [ ] Add criterion benchmarks for:
  - SentencePiece tokenization
  - BPE tokenization (single-pattern)
  - BPE tokenization (multi-pattern)
  - Batch encoding
  - Decode operations
- [ ] Establish baseline performance metrics
- [ ] Document expected performance characteristics

### 3.2 Fuzzing (3 points)
- [ ] Set up cargo-fuzz for BPE pre-tokenization
- [ ] Fuzz multi-pattern splitting
- [ ] Fuzz BPE merge algorithm
- [ ] Fuzz encode/decode round-trips

### 3.3 Memory & Safety (2 points)
- [ ] Run with miri to check for undefined behavior
- [ ] Profile memory usage on large inputs
- [ ] Check for leaks with valgrind/instruments
- [ ] Review unsafe code (if any)

## Phase 4: API Polish (5 Fibonacci points)

### 4.1 Error Messages (2 points)
- [ ] Review all error messages for clarity
- [ ] Add context to errors (e.g., which file, which operation)
- [ ] Ensure errors are actionable

### 4.2 API Consistency (2 points)
- [ ] Review method naming consistency
- [ ] Check parameter order conventions
- [ ] Verify builder pattern usage (if applicable)
- [ ] Document stability guarantees

### 4.3 Examples & Docs (1 point)
- [ ] Add examples/ directory with real-world usage
- [ ] Update README with v0.6.0 achievements
- [ ] Add CHANGELOG.md

## Phase 5: Pre-Integration Checklist (3 Fibonacci points)

### 5.1 Integration Readiness (2 points)
- [ ] Document libshimmy integration points
- [ ] List any breaking changes from v0.5.0
- [ ] Create migration guide if needed
- [ ] Tag v0.6.0 release

### 5.2 CI/CD (1 point)
- [ ] Verify GitHub Actions passes
- [ ] Check docs.rs builds correctly
- [ ] Verify crates.io publishing works

## Total Effort: ~37 Fibonacci points

## Success Criteria
- [ ] Zero clippy warnings
- [ ] Zero dead code warnings
- [ ] >80% test coverage (use cargo-tarpaulin)
- [ ] All 6+ model validations at 100%
- [ ] Benchmarks established
- [ ] Fuzzing runs clean for 1hr+
- [ ] Documentation complete
- [ ] Ready for libshimmy integration
