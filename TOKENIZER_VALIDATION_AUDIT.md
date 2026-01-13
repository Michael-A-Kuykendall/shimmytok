# Tokenizer Validation Audit Plan

**Purpose**: Systematically validate all shimmytok tokenizers against llama.cpp ground truth.

**Tools Available**:
- llama.cpp tokenize: `../llama.cpp/build/bin/Release/llama-tokenize.exe`
- Model files: `../llama.cpp/models/ggml-vocab-*.gguf`

**Test Prompts** (use consistently across all tokenizers):
```
PROMPT_1="Hello, world!"
PROMPT_2="The quick brown fox jumps over the lazy dog."
PROMPT_3="Hello\nWorld\tTab"
PROMPT_4="日本語テスト Chinese中文"
PROMPT_5="<|endoftext|>Special tokens test"
```

---

## Phase 1: Identify Tokenizer Types per Model

Run this command for each model to identify its tokenizer type:

```bash
# Check model metadata
../llama.cpp/build/bin/Release/llama-tokenize.exe -m ../llama.cpp/models/MODEL.gguf -p "test" --log-disable 2>&1
```

### Checklist: Model → Tokenizer Type Mapping

- [ ] **Step 1.1**: Run metadata extraction script
  ```bash
  for f in ../llama.cpp/models/ggml-vocab-*.gguf; do
    echo "=== $f ===" 
    ../llama.cpp/build/bin/Release/llama-tokenize.exe -m "$f" -p "test" --ids --log-disable 2>&1 | head -5
  done
  ```

- [ ] **Step 1.2**: Record model types in table below

| Model File | Tokenizer Type | shimmytok Module | Status |
|------------|---------------|------------------|--------|
| ggml-vocab-bert-bge.gguf | WPM | wpm.rs | ⬜ Untested |
| ggml-vocab-gpt-2.gguf | BPE | bpe.rs | ✅ Tested |
| ggml-vocab-llama-spm.gguf | SPM | sentencepiece.rs | ✅ Tested |
| ggml-vocab-llama-bpe.gguf | BPE | bpe.rs | ✅ Tested |
| ggml-vocab-qwen2.gguf | BPE | bpe.rs | ⬜ Untested |
| ggml-vocab-deepseek-coder.gguf | BPE | bpe.rs | ⬜ Untested |
| ggml-vocab-deepseek-llm.gguf | BPE | bpe.rs | ⬜ Untested |
| ggml-vocab-starcoder.gguf | BPE | bpe.rs | ⬜ Untested |
| ggml-vocab-falcon.gguf | BPE | bpe.rs | ⬜ Untested |
| ggml-vocab-phi-3.gguf | SPM | sentencepiece.rs | ⬜ Untested |
| ggml-vocab-command-r.gguf | BPE | bpe.rs | ⬜ Untested |
| ggml-vocab-refact.gguf | BPE | bpe.rs | ⬜ Untested |

---

## Phase 2: Generate Ground Truth from llama.cpp

For each model, generate expected token IDs.

### Checklist: Ground Truth Generation

- [ ] **Step 2.1**: Create ground truth script
  ```bash
  # File: scripts/generate_ground_truth.sh
  TOKENIZE="../llama.cpp/build/bin/Release/llama-tokenize.exe"
  MODELS="../llama.cpp/models"
  
  mkdir -p test_data/ground_truth
  
  # Prompts
  PROMPTS=(
    "Hello, world!"
    "The quick brown fox jumps over the lazy dog."
    "1234567890"
    "日本語テスト"
    "   multiple   spaces   "
  )
  
  for model in $MODELS/ggml-vocab-*.gguf; do
    name=$(basename $model .gguf)
    echo "Processing $name..."
    for i in "${!PROMPTS[@]}"; do
      $TOKENIZE -m "$model" -p "${PROMPTS[$i]}" --ids --no-bos --log-disable 2>/dev/null \
        > "test_data/ground_truth/${name}_prompt${i}.txt"
    done
  done
  ```

- [ ] **Step 2.2**: Run script and verify output files exist

- [ ] **Step 2.3**: Spot-check a few outputs manually

---

## Phase 3: WPM (WordPiece) Validation

Model: `ggml-vocab-bert-bge.gguf`

### Checklist: WPM Validation

- [ ] **Step 3.1**: Get llama.cpp output
  ```bash
  ../llama.cpp/build/bin/Release/llama-tokenize.exe \
    -m ../llama.cpp/models/ggml-vocab-bert-bge.gguf \
    -p "Hello, world!" --ids --no-bos --log-disable
  ```
  Expected output: `[____]` (fill in)

- [ ] **Step 3.2**: Create Rust test
  ```rust
  #[test]
  fn test_wpm_bert_hello_world() {
      let tokenizer = Tokenizer::from_gguf_file("../llama.cpp/models/ggml-vocab-bert-bge.gguf").unwrap();
      let tokens = tokenizer.encode("Hello, world!", false).unwrap();
      assert_eq!(tokens, vec![/* expected from step 3.1 */]);
  }
  ```

- [ ] **Step 3.3**: Run test, fix discrepancies

- [ ] **Step 3.4**: Add edge case tests:
  - [ ] CJK characters
  - [ ] Punctuation
  - [ ] Unknown words

- [ ] **Step 3.5**: WPM marked as ✅ validated

---

## Phase 4: RWKV Validation

**BLOCKER**: Need RWKV model file. Check if available:

### Checklist: RWKV Validation

- [ ] **Step 4.1**: Find RWKV model
  ```bash
  find ../llama.cpp -name "*rwkv*" -o -name "*RWKV*" 2>/dev/null
  ```
  
- [ ] **Step 4.2**: If not found, download from HuggingFace
  ```bash
  # Example: RWKV-4-Pile models converted to GGUF
  ```

- [ ] **Step 4.3**: Generate ground truth (same as Phase 2)

- [ ] **Step 4.4**: Create validation tests

- [ ] **Step 4.5**: RWKV marked as ✅ validated

---

## Phase 5: UGM (Unigram/T5) Validation

**BLOCKER**: Need T5 or UGM model file.

### Checklist: UGM Validation

- [ ] **Step 5.1**: Find T5/UGM model
  ```bash
  # T5 models use UGM tokenizer
  find .. -name "*t5*" -o -name "*flan*" 2>/dev/null | grep -i gguf
  ```

- [ ] **Step 5.2**: If not found, convert T5 to GGUF or download

- [ ] **Step 5.3**: Generate ground truth

- [ ] **Step 5.4**: Create validation tests
  - [ ] Basic text
  - [ ] User-defined tokens (new preprocessing code)
  - [ ] Unknown token handling

- [ ] **Step 5.5**: UGM marked as ✅ validated

---

## Phase 6: PLaMo-2 Validation

**BLOCKER**: Need PLaMo-2 model file.

### Checklist: PLaMo-2 Validation

- [ ] **Step 6.1**: Find PLaMo-2 model
  ```bash
  find .. -name "*plamo*" 2>/dev/null | grep -i gguf
  ```

- [ ] **Step 6.2**: If not found, download from pfnet

- [ ] **Step 6.3**: Generate ground truth

- [ ] **Step 6.4**: Create validation tests

- [ ] **Step 6.5**: PLaMo-2 marked as ✅ validated

---

## Phase 7: Expand BPE Coverage

Already have basic BPE tests. Expand to cover all patterns.

### Checklist: BPE Multi-Pattern Validation

- [ ] **Step 7.1**: DeepSeek-Coder (5 patterns)
  ```bash
  ../llama.cpp/build/bin/Release/llama-tokenize.exe \
    -m ../llama.cpp/models/ggml-vocab-deepseek-coder.gguf \
    -p "def hello():\n    print('world')" --ids --no-bos --log-disable
  ```

- [ ] **Step 7.2**: DeepSeek-LLM (6 patterns)

- [ ] **Step 7.3**: StarCoder (2 patterns)

- [ ] **Step 7.4**: Qwen2

- [ ] **Step 7.5**: Falcon

- [ ] **Step 7.6**: Command-R

- [ ] **Step 7.7**: All BPE variants marked as ✅ validated

---

## Phase 8: Create Comprehensive Test Suite

### Checklist: Test File Creation

- [ ] **Step 8.1**: Create `tests/test_llama_cpp_validation.rs`
  - Include all ground truth comparisons
  - Use `#[ignore]` for tests requiring model files
  - Document model file paths

- [ ] **Step 8.2**: Add CI configuration (optional)
  - Download model files in CI
  - Run validation tests

- [ ] **Step 8.3**: Update README with validation status

---

## Final Checklist

- [ ] SPM (SentencePiece) - ✅ Already validated
- [ ] BPE (GPT-2 style) - ✅ Already validated  
- [ ] WPM (WordPiece/BERT) - ⬜ Needs validation
- [ ] RWKV (Trie-based) - ⬜ Needs model file + validation
- [ ] UGM (Unigram/T5) - ⬜ Needs model file + validation
- [ ] PLaMo-2 (Table-driven) - ⬜ Needs model file + validation

---

## Notes

### How to run llama.cpp tokenize
```bash
# Basic usage
../llama.cpp/build/bin/Release/llama-tokenize.exe -m MODEL.gguf -p "text" --ids --no-bos --log-disable

# Options:
#   --ids          Output only token IDs as [1, 2, 3]
#   --no-bos       Don't add BOS token (for comparison)
#   --log-disable  Suppress model loading logs
```

### Model file locations
- llama.cpp vocab files: `../llama.cpp/models/ggml-vocab-*.gguf`
- Full models: Check HuggingFace or local cache

### shimmytok test command
```bash
cargo test --test test_llama_cpp_validation -- --nocapture
```
