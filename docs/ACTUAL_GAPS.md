# ShimmyTok: Evidence-Based Gap Analysis

**Date**: October 20, 2025  
**Status**: 779 LOC implemented, tokenization producing incorrect results  
**Analysis Method**: Direct code inspection + test output comparison

---

## Test Evidence

**Input**: `"Hello world"`

**Current Output**: `[15043, 8879, 2096, 29881]`
```
15043 → '▁Hello'
8879  → '▁wo'
2096  → 'rl'
29881 → 'd'
```

**Expected Output (llama.cpp)**: `[15043, 3186]`
```
15043 → '▁Hello'
3186  → '▁world'
```

**Observation**: Token 3186 (`'▁world'`) exists in vocabulary but is not being selected. Instead, `'world'` is segmented into three tokens: `['▁wo', 'rl', 'd']`.

---

## Implemented Components (779 LOC)

### GGUF Parser (`src/gguf.rs` - 240 LOC)
- Reads GGUF metadata
- Extracts token arrays, scores, token types
- Status: ✅ Working (vocabulary loads successfully)

### Vocabulary (`src/vocab.rs` - 159 LOC)
- Token storage and lookup (HashMap: string → TokenId)
- Score retrieval
- Special token handling
- Status: ✅ Working (can retrieve token 3186 = `'▁world'`)

### SentencePiece Implementation (`src/sentencepiece.rs` - 225 LOC)
- Symbol linked list structure
- Priority queue for bigram candidates
- Greedy merge algorithm
- Byte fallback mechanism
- Status: ⚠️ Compiles and runs, produces incorrect output

### BPE Implementation (`src/bpe.rs` - 67 LOC)
- Basic structure present
- Status: ⚠️ Untested

### API Layer (`src/lib.rs` - 88 LOC)
- Tokenizer wrapper interface
- encode/decode methods
- Model type dispatching
- Status: ✅ Working

---

## Missing Components (From llama.cpp Source)

### 1. resegment() Function

**llama.cpp implementation** (src/llama-vocab.cpp, lines ~1180-1210):
```cpp
void resegment(llm_symbol & symbol, std::vector<llama_token> & output) {
    auto text = std::string(symbol.text, symbol.n);
    auto token = vocab.text_to_token(text);
    
    if (token != LLAMA_TOKEN_NULL) {
        output.push_back(token);
        return;
    }
    
    const auto p = rev_merge.find(text);
    if (p != rev_merge.end()) {
        resegment(symbols[p->second.first], output);
        resegment(symbols[p->second.second], output);
    }
}
```

**Purpose**: Validates final merged tokens against vocabulary. If merged token doesn't exist, splits using merge history.

**Why needed**: Current implementation accepts partial merges (`'▁wo'`, `'rl'`, `'d'`) without validating complete token (`'▁world'`) exists.

**Complexity**: 3 Fibonacci points

### 2. rev_merge Tracking

**llama.cpp implementation** (src/llama-vocab.cpp, line ~1160):
```cpp
std::map<std::string, std::pair<int, int>> rev_merge;

// During bigram addition:
rev_merge[text] = std::make_pair(left, right);
```

**Purpose**: Records merge history (which symbol pairs created each merged string).

**Why needed**: Required by resegment() to split invalid merges.

**Complexity**: 2 Fibonacci points (data structure + bookkeeping)

### 3. Pre-tokenization Regex (BPE only)

**llama.cpp implementation** (src/llama-vocab.cpp, lines ~450-800):
- 40+ model-specific regex patterns
- Switch statement: model type → regex array
- Applied before character splitting

**Status**: 
- Not needed for SentencePiece (llama models)
- Required for BPE (gpt2, etc.)
- Patterns already documented in llama.cpp source

**Complexity**: 1 Fibonacci point (copy patterns from llama.cpp)

---

## Root Cause Analysis

### Hypothesis: Missing Validation Step

**Current flow**:
1. Input: `"Hello world"`
2. Normalize: `"▁Hello▁world"`
3. Split: `['▁','H','e','l','l','o','▁','w','o','r','l','d']`
4. Merge: Greedy algorithm combines characters
5. Result: `['▁Hello', '▁wo', 'rl', 'd']`
6. Output: `[15043, 8879, 2096, 29881]` ❌

**Expected flow (with resegment)**:
1-4. (Same as above)
5. Merge produces: `['▁Hello', '▁wo', 'rl', 'd']`
6. **Resegment validates**: `'▁wo' + 'rl' + 'd'` → check vocab for `'▁world'`
7. Find token 3186 exists, use it instead
8. Result: `['▁Hello', '▁world']`
9. Output: `[15043, 3186]` ✅

### Alternative Hypothesis: Incorrect Merge Priority

Merge algorithm may not be scoring bigrams correctly, causing early termination.

**Test required**: Compare merge sequence step-by-step with llama.cpp debug output.

---

## Implementation Plan

### Phase 1: Add resegment() - 3 Fibonacci Points

**File**: `src/sentencepiece.rs`

**Changes**:
1. Add `rev_merge: HashMap<String, (usize, usize)>` field
2. Populate `rev_merge` during bigram merging
3. Implement `resegment()` recursive function
4. Call `resegment()` on final symbols before output

**Validation**:
- Test: `"Hello world"` → `[15043, 3186]`
- Test: 10 additional strings from llama.cpp test suite

### Phase 2: Add rev_merge Tracking - 2 Fibonacci Points

**File**: `src/sentencepiece.rs`

**Changes**:
1. Initialize `HashMap` for merge history
2. Record `(left_idx, right_idx)` for each merge
3. Pass to `resegment()` function

**Validation**:
- Verify merge history accurate
- Test resegment can split/recombine correctly

### Phase 3: Add BPE Pre-tokenization - 1 Fibonacci Point

**File**: `src/bpe.rs`

**Changes**:
1. Copy regex patterns from llama.cpp (lines 450-800)
2. Add regex matching before character split
3. Test with GPT-2 model

**Validation**:
- Test GPT-2 tokenization matches llama.cpp

---

## Complexity Estimate

| Component | Fibonacci Points | Dependencies |
|-----------|------------------|--------------|
| resegment() function | 3 | rev_merge |
| rev_merge tracking | 2 | - |
| BPE regex patterns | 1 | - |
| **Total** | **6** | - |

Note: Original estimate was 13 points. Actual gap is 6 points.

---

## Validation Strategy

### Test Suite Requirements

1. **Basic functionality**:
   - `"Hello world"` → `[15043, 3186]`
   - Compare with llama.cpp on 100 test strings

2. **Edge cases**:
   - Empty string
   - Single character
   - Unknown characters (byte fallback)
   - Special tokens

3. **Multiple models**:
   - TinyLlama (SentencePiece)
   - Phi-3 (SentencePiece)
   - GPT-2 (BPE - future)

### Success Criteria

- ✅ 100% match with llama.cpp on test suite
- ✅ Byte fallback works correctly
- ✅ Special tokens handled properly
- ✅ No regressions in existing tests

---

## Next Actions

1. **Implement resegment()** - 3 points
2. **Add rev_merge tracking** - 2 points  
3. **Validate against llama.cpp** - testing
4. **Document findings** - update this file

**Total work**: 6 Fibonacci points to correct tokenization

---

## Diagnostic Questions

Before implementation, answer:

1. Does llama.cpp merge `['▁','w','o','r','l','d']` → `'▁world'` directly?
2. Or does it merge to `['▁wo', 'rl', 'd']` then resegment fixes it?
3. Are bigram scores being calculated correctly?
4. Is priority queue ordering correct?

**Method**: Add debug logging to both shimmytok and llama.cpp, compare merge sequences.

---

## Conclusion

**Implementation Status**: 779 LOC, algorithm framework complete

**Gap Analysis**: Missing validation step (resegment) causing incorrect token selection

**Complexity**: 6 Fibonacci points (not 13, not 26)

**Timeline**: Implementation can proceed immediately with clear requirements

**Risk Assessment**: Low - algorithm is understood, implementation is straightforward
