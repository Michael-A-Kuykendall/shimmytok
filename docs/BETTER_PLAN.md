# ShimmyTok: Evidence-Based Implementation Plan

## What We Actually Need (From llama.cpp Analysis)

### 1. GGUF Metadata Keys (CONFIRMED)
```
tokenizer.ggml.model              # "llama", "gpt2", "bert", etc.
tokenizer.ggml.pre                # Pre-tokenizer type (40+ variants!)
tokenizer.ggml.tokens             # String array of tokens
tokenizer.ggml.scores             # Float array (for SPM)
tokenizer.ggml.token_type        # Int array of token types
tokenizer.ggml.merges             # String array (for BPE) - DIRECTLY STORED
tokenizer.ggml.add_bos_token     # Bool
tokenizer.ggml.add_eos_token     # Bool
tokenizer.ggml.add_space_prefix  # Bool
tokenizer.ggml.precompiled_charsmap # Binary blob (for some models)
```

### 2. Token Types (From llama.cpp)
```cpp
LLAMA_TOKEN_TYPE_UNDEFINED    = 0
LLAMA_TOKEN_TYPE_NORMAL       = 1  
LLAMA_TOKEN_TYPE_UNKNOWN      = 2
LLAMA_TOKEN_TYPE_CONTROL      = 3
LLAMA_TOKEN_TYPE_USER_DEFINED = 4
LLAMA_TOKEN_TYPE_UNUSED       = 5
LLAMA_TOKEN_TYPE_BYTE         = 6
```

### 3. Pre-Tokenizer Types (40+ variants!)
```cpp
LLAMA_VOCAB_PRE_TYPE_DEFAULT    = 0
LLAMA_VOCAB_PRE_TYPE_LLAMA3     = 1
LLAMA_VOCAB_PRE_TYPE_DEEPSEEK   = 2
LLAMA_VOCAB_PRE_TYPE_GPT2       = 7
... // 40+ more variants
```

## Revised Implementation Strategy

### Week 1: Extract Real GGUF Data
```bash
# 1. Dump actual GGUF files
for model in tinyllama phi3 mistral; do
    gguf-dump $model.gguf --json > ${model}_metadata.json
done

# 2. Run llama.cpp tokenizer on test cases
./llama-tokenize -m model.gguf < test_cases.txt > expected_output.json

# 3. Document EVERY metadata key found
```

### Week 2: Start with SIMPLEST Case
- Implement GPT2 BPE ONLY (merges are stored directly)
- Skip SentencePiece initially (too complex)
- Validate against llama.cpp byte-for-byte

### Week 3: Add Complexity Incrementally
- Add one pre-tokenizer type at a time
- Test against llama.cpp after EACH addition
- Document model-specific quirks

## Critical Implementation Details

### SentencePiece Reality Check
- NOT simple Viterbi - uses precompiled tries
- Whitespace becomes ▁ (special handling)
- Byte fallback for unknown sequences
- Model-specific normalization (NFKC, NFC, none)

### BPE Reality Check
- Merges stored as string pairs in GGUF
- Format: "Ġt he" (with special Ġ prefix)
- No extraction needed - just read the array

### Special Token Detection
llama.cpp auto-detects by pattern matching:
- EOT: "<|eot_id|>", "<|im_end|>", "<end_of_turn>"
- FIM_PREFIX: "<|fim_prefix|>", "<PRE>", etc.

## Minimum Viable Product (REVISED)

### Target: GPT2-style BPE tokenizer ONLY
- Read GGUF metadata correctly
- Load BPE merges from `tokenizer.ggml.merges`
- Apply merges in order
- Handle special tokens
- Match llama.cpp output exactly

### Non-Goals for MVP
- SentencePiece (too complex)
- All 40+ pre-tokenizer types
- Performance optimization
- Unicode edge cases

## Success Metrics (REVISED)

### Week 1 Success
- [ ] Can dump and understand 3+ GGUF files
- [ ] Have test vectors from llama.cpp
- [ ] Understand actual metadata structure

### Week 2 Success  
- [ ] GPT2 BPE tokenizer works
- [ ] Matches llama.cpp on 100 test cases
- [ ] Handles special tokens correctly

### Week 3 Success
- [ ] Add 1-2 more tokenizer types
- [ ] Document limitations clearly
- [ ] Release as v0.0.1 (experimental)

## Risk Assessment (UPDATED)

**Previous confidence**: 75%
**Revised confidence**: 25% for full implementation, 60% for BPE-only MVP

**Why lower?**
- 40+ pre-tokenizer variants (not 2)
- Complex token attribute system
- Model-specific normalization
- Special token auto-detection logic

**Path forward**: 
1. Start with BPE only (achievable)
2. Add features incrementally
3. Be honest about limitations
4. Ship working subset, not broken whole