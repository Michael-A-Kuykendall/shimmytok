# ShimmyTok Hostile Audit - CRITICAL GAPS

## Executive Summary
The SHIMMYTOK_MASTER_PLAN.md is **dangerously underspecified** with multiple fatal assumptions that WILL cause failure. This document identifies every critical gap and provides the empirical approach to fix them.

## FATAL FLAWS IDENTIFIED

### 1. GGUF Vocabulary Format - COMPLETELY UNKNOWN
**Problem**: Plan assumes metadata keys without verification
**Reality**: GGUF format varies significantly between models
**Required Action**: 
```bash
# Extract actual metadata from real models
for model in *.gguf; do
    echo "=== $model ==="
    python3 -c "import struct; f=open('$model','rb'); 
    # Parse GGUF header and dump ALL metadata keys
done
```

### 2. SentencePiece Algorithm - TOO SIMPLISTIC
**Problem**: Basic Viterbi won't match llama.cpp
**Missing**:
- Trie/automaton for O(1) lookups (not O(V) scanning)
- Whitespace → ▁ transformation rules
- Byte-fallback for unknown sequences
- NFKC normalization (not NFC)
- Score interpretation (log vs linear)

**Required**: Study llama.cpp's actual implementation:
```cpp
// From llama.cpp - unicode.cpp
struct llm_tokenizer_spm {
    llm_symbol_table symbols; // Trie structure
    llm_normalize_map normalizer; // Custom normalization
    // ... complex state machines
};
```

### 3. BPE Merge Extraction - IMPOSSIBLE AS DESCRIBED
**Problem**: Cannot reconstruct merges from vocabulary
**Reality**: Merges stored separately in GGUF
**Fix**: Read `tokenizer.ggml.merges` array directly

### 4. Token Type Semantics - MISUNDERSTOOD
Token types have MODEL-SPECIFIC meanings:
- Type 1 (NORMAL): Regular token
- Type 2 (UNKNOWN): Model-specific
- Type 3 (CONTROL): BOS/EOS/SEP/PAD 
- Type 6 (BYTE): Fallback bytes 0x00-0xFF

### 5. Special Token Logic - MISSING CRITICAL FLAGS
**Required metadata** (not mentioned in plan):
- `tokenizer.ggml.add_bos_token`
- `tokenizer.ggml.add_eos_token`
- `tokenizer.ggml.add_space_prefix`
- `tokenizer.ggml.clean_up_tokenization_spaces`

## EMPIRICAL DATA EXTRACTION PLAN

### Phase 1: GGUF Format Discovery
```bash
# 1. Dump raw GGUF structure
hexdump -C model.gguf | head -n 1000 > gguf_structure.txt

# 2. Extract with existing parser
gguf-dump model.gguf --json > metadata.json

# 3. Compare multiple models
for model in tinyllama phi3 mistral gpt2; do
    gguf-dump $model.gguf --json > ${model}_metadata.json
done
diff -u tinyllama_metadata.json phi3_metadata.json
```

### Phase 2: llama.cpp Source Analysis
```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp

# Key files to study:
# - llama.cpp (lines 13000-15000 for tokenization)
# - unicode.cpp/h (normalization tables)
# - common/common.cpp (tokenizer initialization)

# Extract tokenizer test cases
grep -r "test.*tokenize" tests/
```

### Phase 3: Generate Ground Truth
```bash
# Build llama.cpp tools
make llama-tokenize

# Generate test vectors
cat > test_inputs.txt << EOF
Hello world
The quick brown fox
émoji 👍🏽 test
<|im_start|>system
EOF

# Tokenize with llama.cpp
while read line; do
    echo "Input: $line"
    ./llama-tokenize -m model.gguf "$line"
done < test_inputs.txt > ground_truth.json
```

### Phase 4: Build Hostile Test Suite
```python
# hostile_tests.py
test_cases = [
    # Unicode edge cases
    "\x00\x01\x02",  # Control characters
    "𝕳𝖊𝖑𝖑𝖔",  # Mathematical alphanumeric
    "🏴󠁧󠁢󠁳󠁣󠁴󠁿",  # Flag sequence
    "\u200b\ufeff",  # Zero-width spaces
    
    # Tokenizer attacks
    " " * 10000,  # Pathological whitespace
    "a" * 100000,  # Repetition attack
    "<|" + "a" * 1000 + "|>",  # Special token injection
    
    # Encoding issues
    b"\xff\xfe".decode('latin1'),  # Invalid UTF-8
    "A" + chr(0x0301) * 100,  # Combining character spam
]
```

## CORRECTED IMPLEMENTATION APPROACH

### Week 1: EMPIRICAL GGUF Analysis
- [ ] Dump 5+ different GGUF files
- [ ] Document EVERY metadata key found
- [ ] Map metadata to tokenizer behavior
- [ ] Build GGUF reader that handles ALL variants

### Week 2: llama.cpp Reverse Engineering
- [ ] Extract tokenizer implementation
- [ ] Document ACTUAL algorithm (not paper)
- [ ] Identify ALL special cases
- [ ] Build test harness against llama.cpp

### Week 3: Incremental Implementation
- [ ] Start with SIMPLEST model (GPT-2 BPE)
- [ ] Validate EVERY line against llama.cpp
- [ ] Add models one at a time
- [ ] Document model-specific quirks

### Week 4: Hostile Testing
- [ ] Fuzz with malformed inputs
- [ ] Benchmark against llama.cpp
- [ ] Memory safety validation
- [ ] Edge case coverage

## SUCCESS CRITERIA (REVISED)

### Minimum Viable
- [ ] Matches llama.cpp EXACTLY on 100 test cases
- [ ] Handles TinyLlama model correctly
- [ ] No crashes on malformed input
- [ ] Performance within 2x of llama.cpp

### Production Ready
- [ ] 1000+ test cases passing
- [ ] 5+ model architectures supported
- [ ] Fuzz testing for 24 hours without crashes
- [ ] Performance within 1.5x of llama.cpp

## CONFIDENCE ASSESSMENT (CORRECTED)

**Original**: 75% confidence
**Revised**: 40% confidence

**Why lower?**
- GGUF format more complex than assumed
- SentencePiece has undocumented behaviors  
- BPE merge extraction was completely wrong
- Performance requirements underestimated

**Path to 75% confidence:**
1. Complete empirical GGUF analysis (raises to 50%)
2. Successfully match llama.cpp on one model (60%)
3. Pass hostile test suite (70%)
4. Support 3+ models correctly (75%)

## CONCLUSION

This plan needs MAJOR revision before implementation. The core algorithms are more complex than described, the GGUF format is underspecified, and the testing strategy is insufficient. 

**Recommendation**: Spend 1 week on empirical analysis BEFORE writing any implementation code.