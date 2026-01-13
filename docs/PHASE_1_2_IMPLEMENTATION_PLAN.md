# Phase 1 & 2 Implementation Plan

**Status**: Ready to Execute  
**Target**: Complete both phases in 2-3 days  
**Scope**: Performance optimization + Model expansion  
**Version Target**: v0.2.0 → v0.3.0

---

## Overview

This document breaks down ROADMAP.md Phase 1 (Performance) and Phase 2 (Model Support) into executable steps with validation gates. Each implementation step is followed by a validation step to ensure correctness and non-breaking changes.

**Execution Model**:
- ✅ Step N: Implement feature
- 🧪 Step N+1: Validate feature (tests + benchmarks)
- Repeat until complete

---

## Phase 1: Performance Optimization (v0.2.0)

**Goal**: 2-5x speedup without breaking API  
**Estimated Effort**: 8 Fibonacci points

### Step 1: Set Up Benchmark Suite
**Task**: Create benchmarking infrastructure

**Actions**:
1. Create `benches/tokenization.rs` with criterion setup
2. Add benchmark for `encode()` throughput
3. Add benchmark for `decode()` throughput
4. Add benchmark for `from_gguf_file()` load time
5. Baseline current performance

**Files**:
```
benches/
  tokenization.rs
Cargo.toml (add [dev-dependencies] criterion)
```

**Code**:
```rust
// benches/tokenization.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use shimmytok::Tokenizer;

fn bench_encode(c: &mut Criterion) {
    let tokenizer = Tokenizer::from_gguf_file("~/.cache/models/gguf/gpt2.Q4_K_M.gguf")
        .expect("Failed to load");
    
    let mut group = c.benchmark_group("encode");
    
    for size in [10, 100, 1000, 10000].iter() {
        let text = "Hello world ".repeat(*size);
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| tokenizer.encode(black_box(&text), false));
        });
    }
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let tokenizer = Tokenizer::from_gguf_file("~/.cache/models/gguf/gpt2.Q4_K_M.gguf")
        .expect("Failed to load");
    
    let tokens: Vec<u32> = (0..1000).map(|i| i % tokenizer.vocab_size() as u32).collect();
    
    c.bench_function("decode_1000_tokens", |b| {
        b.iter(|| tokenizer.decode(black_box(&tokens), false));
    });
}

fn bench_load(c: &mut Criterion) {
    c.bench_function("load_tokenizer", |b| {
        b.iter(|| Tokenizer::from_gguf_file("~/.cache/models/gguf/gpt2.Q4_K_M.gguf"));
    });
}

criterion_group!(benches, bench_encode, bench_decode, bench_load);
criterion_main!(benches);
```

**Validation Criteria**:
- Benchmarks compile
- Baseline numbers recorded
- Can run `cargo bench`

---

### Step 2: Validate Benchmark Suite
**Task**: Ensure benchmarks work and establish baseline

**Actions**:
1. Run `cargo bench` and record baseline
2. Verify benchmarks complete without errors
3. Document baseline performance in `internal_docs/BASELINE_PERFORMANCE.md`
4. Check benchmarks run in CI (optional)

**Success Criteria**:
- ✅ All benchmarks run
- ✅ Baseline documented
- ✅ No panics or errors

---

### Step 3: Implement Vocabulary Caching
**Task**: Add HashMap cache for frequent token lookups

**Actions**:
1. Add `HashMap<String, TokenId>` to `Vocabulary` struct
2. Populate cache during `from_gguf_file()`
3. Use cache in lookup paths (piece → token ID)
4. Ensure cache is transparent (no API changes)

**Files**:
```
src/vocab.rs
```

**Code Strategy**:
```rust
// In src/vocab.rs
pub struct Vocabulary {
    // Existing fields...
    piece_to_id: HashMap<String, TokenId>, // NEW: cache
}

impl Vocabulary {
    pub fn from_gguf_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        // ... existing parsing ...
        
        // NEW: Build cache
        let mut piece_to_id = HashMap::new();
        for (id, piece) in tokens.iter().enumerate() {
            piece_to_id.insert(piece.clone(), id as TokenId);
        }
        
        Ok(Self {
            // ... existing fields ...
            piece_to_id,
        })
    }
    
    // Use cache in lookups
    pub fn token_to_id(&self, piece: &str) -> Option<TokenId> {
        self.piece_to_id.get(piece).copied()
    }
}
```

**Validation Criteria**:
- Vocabulary creation still works
- Lookups use cache
- No memory leaks

---

### Step 4: Validate Vocabulary Caching
**Task**: Test correctness and measure speedup

**Actions**:
1. Run full test suite: `cargo test`
2. Run benchmarks: `cargo bench`
3. Compare against baseline (expect 1.5-2x for encode)
4. Check memory usage (should be reasonable)

**Success Criteria**:
- ✅ All tests pass
- ✅ 1.5-2x speedup on encode
- ✅ Memory increase <20%

---

### Step 5: Implement Parallel Batch Encoding
**Task**: Add `encode_batch()` method using rayon

**Actions**:
1. Add rayon to dependencies
2. Implement `encode_batch()` in `src/lib.rs`
3. Add doc comments and examples
4. Keep single `encode()` unchanged

**Files**:
```
Cargo.toml (add rayon)
src/lib.rs
```

**Code**:
```rust
// In Cargo.toml
[dependencies]
rayon = "1.10"

// In src/lib.rs
use rayon::prelude::*;

impl Tokenizer {
    /// Encode multiple texts in parallel
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use shimmytok::Tokenizer;
    /// # let tokenizer = Tokenizer::from_gguf_file("model.gguf")?;
    /// let texts = vec!["Hello world", "Goodbye world"];
    /// let batch = tokenizer.encode_batch(&texts, true)?;
    /// # Ok::<(), shimmytok::Error>(())
    /// ```
    pub fn encode_batch(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> Result<Vec<Vec<TokenId>>, Error> {
        texts
            .par_iter()
            .map(|text| self.encode(text, add_special_tokens))
            .collect()
    }
}
```

**Validation Criteria**:
- New method compiles
- Non-breaking (old API unchanged)
- Documentation complete

---

### Step 6: Validate Parallel Batch Encoding
**Task**: Test correctness and parallel speedup

**Actions**:
1. Add test for `encode_batch()` in `tests/test_basic.rs`
2. Verify output matches sequential encoding
3. Benchmark batch vs sequential
4. Run full test suite

**Test Code**:
```rust
#[test]
fn test_encode_batch() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test_encode_batch: model not found");
        return;
    }
    
    let tokenizer = Tokenizer::from_gguf_file(&model_path).unwrap();
    
    let texts = vec!["Hello world", "Goodbye world", "Rust is great"];
    let batch = tokenizer.encode_batch(&texts, false).unwrap();
    
    // Verify matches sequential
    for (i, text) in texts.iter().enumerate() {
        let sequential = tokenizer.encode(text, false).unwrap();
        assert_eq!(batch[i], sequential, "Batch mismatch for: {}", text);
    }
}
```

**Success Criteria**:
- ✅ Test passes
- ✅ Batch output == sequential output
- ✅ 2-4x speedup on large batches

---

### Step 7: Add Benchmark for Batch Encoding
**Task**: Measure parallel performance gains

**Actions**:
1. Add batch benchmark to `benches/tokenization.rs`
2. Test batch sizes: 1, 10, 100, 1000 texts
3. Compare to sequential encoding
4. Document results

**Code**:
```rust
fn bench_encode_batch(c: &mut Criterion) {
    let tokenizer = Tokenizer::from_gguf_file("~/.cache/models/gguf/gpt2.Q4_K_M.gguf")
        .expect("Failed to load");
    
    let mut group = c.benchmark_group("encode_batch");
    
    for batch_size in [1, 10, 100, 1000].iter() {
        let texts: Vec<String> = (0..*batch_size)
            .map(|i| format!("This is test string number {}", i))
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, _| {
                b.iter(|| tokenizer.encode_batch(black_box(&text_refs), false));
            },
        );
    }
    group.finish();
}
```

**Success Criteria**:
- ✅ Benchmark runs
- ✅ Linear scaling up to CPU cores
- ✅ No performance regression

---

### Step 8: Validate Phase 1 Complete
**Task**: Final validation of all Phase 1 work

**Actions**:
1. Run full test suite: `cargo test --all-features`
2. Run all benchmarks: `cargo bench`
3. Check clippy: `cargo clippy -- -D warnings`
4. Check formatting: `cargo fmt -- --check`
5. Update CHANGELOG.md with v0.2.0 notes
6. Verify API is non-breaking (no semver issues)

**Changelog Entry**:
```markdown
## [0.2.0] - 2025-10-XX

### Added
- `encode_batch()` method for parallel encoding of multiple texts
- Comprehensive benchmark suite for performance tracking

### Performance
- 1.5-2x speedup on `encode()` via vocabulary caching
- 2-4x speedup on batch encoding via parallel processing
- Optimized token lookup paths

### Internal
- Added HashMap cache for piece → token ID lookups
- Added rayon for parallel processing
```

**Success Criteria**:
- ✅ All tests pass (30/30)
- ✅ No clippy warnings
- ✅ Formatted correctly
- ✅ Performance targets met
- ✅ API unchanged (libshimmy compatible)

---

## Phase 2: Model Support Expansion (v0.3.0)

**Goal**: Support Mistral, Qwen, Gemma tokenizers  
**Estimated Effort**: 13 Fibonacci points

### Step 9: Research Mistral Tokenizer Format
**Task**: Understand Mistral tokenization in GGUF

**Actions**:
1. Download Mistral GGUF model
2. Use `extract_gguf.py` to inspect metadata
3. Document tokenizer type and configuration
4. Identify differences from "llama" type
5. Check if SentencePiece or new algorithm needed

**Files**:
```
internal_docs/MISTRAL_TOKENIZER_RESEARCH.md
```

**Research Questions**:
- What is `tokenizer.ggml.model` value?
- Are tokens/scores/types same format as LLaMA?
- Any special configurations?
- Pre-tokenizer pattern differences?

**Validation Criteria**:
- Documentation complete
- Test model downloaded
- Ready to implement

---

### Step 10: Implement Mistral Tokenizer Support
**Task**: Add "mistral" model type

**Actions**:
1. Add "mistral" case to `from_gguf_file()` match
2. Determine if SentencePiece or custom implementation
3. Add Mistral-specific configuration handling
4. Update supported models list in README

**Code**:
```rust
// In src/lib.rs
impl Tokenizer {
    pub fn from_gguf_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let vocab = Vocabulary::from_gguf_file(path)?;

        let tokenizer_impl: Box<dyn TokenizerImpl> = match vocab.model_type() {
            "llama" => Box::new(sentencepiece::SentencePieceTokenizer::new()),
            "gpt2" => Box::new(bpe::BPETokenizer::new()),
            "mistral" => Box::new(sentencepiece::SentencePieceTokenizer::new()), // NEW
            model => return Err(Error::UnsupportedModel(model.to_string())),
        };
        // ...
    }
}
```

**Validation Criteria**:
- Code compiles
- Mistral model type recognized
- No breaking changes

---

### Step 11: Validate Mistral Tokenizer
**Task**: Test against Mistral reference output

**Actions**:
1. Get reference tokenization from llama.cpp
2. Create test in `tests/test_mistral.rs`
3. Validate against reference for 5-10 test cases
4. Ensure Unicode handling works

**Test Code**:
```rust
#[test]
fn test_mistral_tokenization() {
    let model_path = "~/.cache/models/gguf/mistral-7b.Q4_K_M.gguf";
    if !Path::new(model_path).exists() {
        eprintln!("Skipping: Mistral model not found");
        return;
    }
    
    let tokenizer = Tokenizer::from_gguf_file(model_path).unwrap();
    
    // Reference from llama.cpp
    let test_cases = vec![
        ("Hello world", vec![1, 22557, 3186]),
        ("Mistral AI", vec![1, 28755, 1650, 16107]),
    ];
    
    for (text, expected) in test_cases {
        let tokens = tokenizer.encode(text, true).unwrap();
        assert_eq!(tokens, expected, "Mismatch for: {}", text);
    }
}
```

**Success Criteria**:
- ✅ Tests pass
- ✅ Matches llama.cpp output
- ✅ Unicode works correctly

---

### Step 12: Research Qwen Tokenizer Format
**Task**: Understand Qwen tokenization in GGUF

**Actions**:
1. Download Qwen GGUF model
2. Use `extract_gguf.py` to inspect metadata
3. Document tokenizer type and configuration
4. Check for BPE vs SentencePiece vs custom
5. Identify any special handling needed

**Files**:
```
internal_docs/QWEN_TOKENIZER_RESEARCH.md
```

**Research Questions**:
- What is `tokenizer.ggml.model` value?
- Token format (BPE-style or SPM-style)?
- Pre-tokenizer pattern (regex)?
- Special tokens handling?

**Validation Criteria**:
- Documentation complete
- Algorithm identified
- Ready to implement

---

### Step 13: Implement Qwen Tokenizer Support
**Task**: Add "qwen" or "qwen2" model type

**Actions**:
1. Add model type case to match statement
2. Implement algorithm (likely BPE variant)
3. Handle Qwen-specific regex patterns
4. Add configuration parsing

**Code Strategy**:
```rust
// If Qwen uses BPE with custom pattern
match vocab.model_type() {
    "llama" => Box::new(sentencepiece::SentencePieceTokenizer::new()),
    "gpt2" => Box::new(bpe::BPETokenizer::new()),
    "mistral" => Box::new(sentencepiece::SentencePieceTokenizer::new()),
    "qwen" | "qwen2" => Box::new(bpe::BPETokenizer::new()), // NEW
    model => return Err(Error::UnsupportedModel(model.to_string())),
};
```

**Validation Criteria**:
- Compiles successfully
- Qwen models load
- No API changes

---

### Step 14: Validate Qwen Tokenizer
**Task**: Test against Qwen reference output

**Actions**:
1. Get reference from llama.cpp or transformers
2. Create `tests/test_qwen.rs`
3. Test multiple text samples
4. Verify CJK handling (important for Qwen)

**Test Code**:
```rust
#[test]
fn test_qwen_tokenization() {
    let model_path = "~/.cache/models/gguf/qwen2.Q4_K_M.gguf";
    if !Path::new(model_path).exists() {
        eprintln!("Skipping: Qwen model not found");
        return;
    }
    
    let tokenizer = Tokenizer::from_gguf_file(model_path).unwrap();
    
    // Test English and Chinese
    let test_cases = vec![
        ("Hello world", vec![/* ref tokens */]),
        ("你好世界", vec![/* ref tokens */]),
        ("Mixed 中文 text", vec![/* ref tokens */]),
    ];
    
    for (text, expected) in test_cases {
        let tokens = tokenizer.encode(text, true).unwrap();
        assert_eq!(tokens, expected, "Mismatch for: {}", text);
    }
}
```

**Success Criteria**:
- ✅ Tests pass
- ✅ CJK tokenization correct
- ✅ Matches reference implementation

---

### Step 15: Research Gemma Tokenizer Format
**Task**: Understand Gemma tokenization in GGUF

**Actions**:
1. Download Gemma GGUF model
2. Inspect with `extract_gguf.py`
3. Document tokenizer configuration
4. Identify algorithm (likely SentencePiece)
5. Check for Google-specific features

**Files**:
```
internal_docs/GEMMA_TOKENIZER_RESEARCH.md
```

**Research Questions**:
- Model type identifier?
- SentencePiece variant?
- Special token handling differences?
- Any Google-specific extensions?

**Validation Criteria**:
- Research complete
- Algorithm determined
- Ready to code

---

### Step 16: Implement Gemma Tokenizer Support
**Task**: Add "gemma" model type

**Actions**:
1. Add "gemma" case to model type match
2. Use appropriate algorithm (likely SentencePiece)
3. Handle Gemma-specific configuration
4. Test loading succeeds

**Code**:
```rust
match vocab.model_type() {
    "llama" => Box::new(sentencepiece::SentencePieceTokenizer::new()),
    "gpt2" => Box::new(bpe::BPETokenizer::new()),
    "mistral" => Box::new(sentencepiece::SentencePieceTokenizer::new()),
    "qwen" | "qwen2" => Box::new(bpe::BPETokenizer::new()),
    "gemma" => Box::new(sentencepiece::SentencePieceTokenizer::new()), // NEW
    model => return Err(Error::UnsupportedModel(model.to_string())),
};
```

**Validation Criteria**:
- Compiles
- Gemma models recognized
- No breaking changes

---

### Step 17: Validate Gemma Tokenizer
**Task**: Test against Gemma reference output

**Actions**:
1. Get reference output from transformers or llama.cpp
2. Create `tests/test_gemma.rs`
3. Test various inputs
4. Verify special token handling

**Test Code**:
```rust
#[test]
fn test_gemma_tokenization() {
    let model_path = "~/.cache/models/gguf/gemma-2b.Q4_K_M.gguf";
    if !Path::new(model_path).exists() {
        eprintln!("Skipping: Gemma model not found");
        return;
    }
    
    let tokenizer = Tokenizer::from_gguf_file(model_path).unwrap();
    
    let test_cases = vec![
        ("Hello world", vec![/* ref tokens */]),
        ("Gemma is a Google model", vec![/* ref tokens */]),
    ];
    
    for (text, expected) in test_cases {
        let tokens = tokenizer.encode(text, true).unwrap();
        assert_eq!(tokens, expected, "Mismatch for: {}", text);
    }
}
```

**Success Criteria**:
- ✅ Tests pass
- ✅ Output matches reference
- ✅ Special tokens handled correctly

---

### Step 18: Add Model Type Detection
**Task**: Implement `model_type()` public method

**Actions**:
1. Add `model_type()` to public API
2. Return string from vocabulary
3. Add doc comments
4. Update examples

**Code**:
```rust
impl Tokenizer {
    /// Get the tokenizer model type
    ///
    /// Returns the model type identifier from the GGUF metadata,
    /// such as "llama", "gpt2", "mistral", "qwen", or "gemma".
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use shimmytok::Tokenizer;
    /// # let tokenizer = Tokenizer::from_gguf_file("model.gguf")?;
    /// println!("Model type: {}", tokenizer.model_type());
    /// # Ok::<(), shimmytok::Error>(())
    /// ```
    pub fn model_type(&self) -> &str {
        self.vocab.model_type()
    }
}
```

**Validation Criteria**:
- Method works for all model types
- Documentation clear
- Non-breaking addition

---

### Step 19: Update Documentation
**Task**: Document new model support

**Actions**:
1. Update README.md supported models table
2. Add examples for new models
3. Update crate-level docs in lib.rs
4. Update CHANGELOG.md for v0.3.0

**README Update**:
```markdown
## Supported Models

| Model Type | Status | Implementation |
|------------|--------|----------------|
| LLaMA/Llama-2/Llama-3 | ✅ Full support | SentencePiece (validated) |
| Phi-3 | ✅ Full support | SentencePiece (validated) |
| Mistral | ✅ Full support | SentencePiece (validated) |
| Qwen/Qwen2 | ✅ Full support | BPE (validated) |
| Gemma | ✅ Full support | SentencePiece (validated) |
| GPT-2 / GPT-3 (BPE) | ✅ Full support | Priority queue BPE |
```

**Changelog**:
```markdown
## [0.3.0] - 2025-10-XX

### Added
- Support for Mistral models (SentencePiece)
- Support for Qwen/Qwen2 models (BPE)
- Support for Gemma models (SentencePiece)
- `model_type()` method to query tokenizer type

### Validated
- Mistral tokenization matches llama.cpp
- Qwen tokenization matches reference (including CJK)
- Gemma tokenization matches reference
```

**Success Criteria**:
- Documentation complete
- Examples accurate
- Changelog updated

---

### Step 20: Validate Phase 2 Complete
**Task**: Final validation of all Phase 2 work

**Actions**:
1. Run full test suite: `cargo test --all-features`
2. Test all model types (LLaMA, GPT-2, Mistral, Qwen, Gemma)
3. Run benchmarks to ensure no regression
4. Check clippy: `cargo clippy -- -D warnings`
5. Check formatting: `cargo fmt -- --check`
6. Verify API compatibility (no breaking changes)
7. Update version in Cargo.toml to 0.3.0

**Success Criteria**:
- ✅ All tests pass (35+ tests)
- ✅ All 5 model types validated
- ✅ No performance regression
- ✅ No clippy warnings
- ✅ Formatted correctly
- ✅ API unchanged (semver minor bump)

---

## Publishing Plan

### Step 21: Publish v0.2.0
**Task**: Release performance improvements

**Actions**:
1. Create git tag: `git tag -a v0.2.0 -m "Performance optimization release"`
2. Push tag: `git push origin v0.2.0`
3. Wait for CI to publish to crates.io
4. Verify on https://crates.io/crates/shimmytok
5. Create GitHub Release with CHANGELOG notes

**Success Criteria**:
- ✅ v0.2.0 live on crates.io
- ✅ GitHub Release created
- ✅ CI workflow succeeded

---

### Step 22: Publish v0.3.0
**Task**: Release model expansion

**Actions**:
1. Create git tag: `git tag -a v0.3.0 -m "Model support expansion"`
2. Push tag: `git push origin v0.3.0`
3. Wait for CI to publish to crates.io
4. Verify on crates.io
5. Create GitHub Release
6. Update shimmy announcement with new model support

**Success Criteria**:
- ✅ v0.3.0 live on crates.io
- ✅ GitHub Release created
- ✅ Announcement updated

---

## Risk Mitigation

### Potential Issues

**Issue 1: Model files not available**
- **Solution**: Skip tests gracefully (already implemented)
- **Workaround**: Use smaller test models from HuggingFace

**Issue 2: Reference outputs don't match**
- **Solution**: Debug with llama.cpp side-by-side
- **Workaround**: Document differences if unavoidable

**Issue 3: Performance targets not met**
- **Solution**: Profile and optimize hot paths
- **Fallback**: Ship what we have, iterate later

**Issue 4: New dependencies break WASM**
- **Solution**: Use feature flags for rayon
- **Workaround**: Make parallel encoding optional

---

## Success Metrics

### Phase 1 Targets
- [x] Benchmark suite created
- [ ] 1.5-2x encode speedup (vocabulary caching)
- [ ] 2-4x batch speedup (parallel processing)
- [ ] `encode_batch()` API added
- [ ] All tests pass
- [ ] No API breaking changes

### Phase 2 Targets
- [ ] Mistral support validated
- [ ] Qwen support validated
- [ ] Gemma support validated
- [ ] `model_type()` API added
- [ ] 5 model types total
- [ ] All tests pass
- [ ] No API breaking changes

---

## Timeline

**Day 1**: Steps 1-8 (Phase 1)
- Morning: Benchmark suite (Steps 1-2)
- Afternoon: Vocabulary caching (Steps 3-4)
- Evening: Batch encoding (Steps 5-7)
- Night: Phase 1 validation (Step 8)

**Day 2**: Steps 9-15 (Phase 2 Part 1)
- Morning: Mistral research + implementation (Steps 9-11)
- Afternoon: Qwen research + implementation (Steps 12-14)
- Evening: Gemma research (Step 15)

**Day 3**: Steps 16-22 (Phase 2 Part 2 + Publishing)
- Morning: Gemma implementation + validation (Steps 16-17)
- Afternoon: Model type detection + docs (Steps 18-19)
- Evening: Phase 2 validation (Step 20)
- Night: Publishing (Steps 21-22)

---

## Audit Results

### ✅ Completeness Check
- [x] All Phase 1 tasks broken down
- [x] All Phase 2 tasks broken down
- [x] Validation steps after each implementation
- [x] Code examples provided
- [x] Test examples provided
- [x] Success criteria defined
- [x] Timeline estimated

### ✅ Correctness Check
- [x] Steps are sequential and logical
- [x] No circular dependencies
- [x] Each step has clear deliverables
- [x] Validation gates prevent moving forward with bugs
- [x] Non-breaking changes verified at each step

### ✅ Feasibility Check
- [x] 2-3 day timeline realistic (8 + 13 = 21 Fibonacci points / 3 days = 7 per day)
- [x] Steps are appropriately sized (not too big, not too small)
- [x] Research steps precede implementation
- [x] Testing integrated throughout

### ✅ Risk Coverage
- [x] Missing models handled (skip tests)
- [x] Reference mismatches have debug plan
- [x] Performance fallback defined
- [x] Dependency issues addressed

### 🎯 Ready to Execute

This plan is **complete, validated, and ready for execution**. Each step builds on the previous, validation gates prevent bugs from propagating, and the timeline is achievable.

---

**Plan Created**: October 22, 2025  
**Status**: READY TO BLAST 🚀  
**Let's go!**
