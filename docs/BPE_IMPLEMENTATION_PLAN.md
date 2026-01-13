# BPE Implementation Plan

**Date**: October 21, 2025  
**Goal**: Implement proper BPE tokenization in pure Rust based on llama.cpp  
**Timeline**: Immediate implementation (Fibonacci 8 = ~6 hours)

---

## What llama.cpp Actually Does (From Source Analysis)

### 1. Pre-Tokenization with Regex (CONFIRMED)

**Per model-type regex patterns** - llama.cpp has 40+ different patterns:

```cpp
// GPT-2 pattern (simplest):
"'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"

// Llama-3 pattern:
"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
```

**Our approach**: 
- Support GPT-2 pattern first (most common)
- Add Llama-3 pattern (for Llama-3 models)
- Make patterns configurable via `pre_type` from GGUF

### 2. BPE Merge Algorithm (Priority-Based)

```cpp
// From llama.cpp llm_tokenizer_bpe_session::tokenize()
for (each word fragment from regex) {
    // Split into UTF-8 characters
    while (offset < word.size()) {
        size_t char_len = unicode_len_utf8(word[offset]);
        symbols.push_back({text: &word[offset], len: char_len});
        offset += char_len;
    }
    
    // Find all possible merges
    for (i in 0..symbols.len()-1) {
        add_new_bigram(i, i+1);  // checks if merge exists in bpe_ranks
    }
    
    // Apply merges in rank order (priority queue)
    while (!work_queue.empty()) {
        bigram = work_queue.pop();  // Highest priority (lowest rank)
        
        // Merge left + right
        left_symbol.len += right_symbol.len;
        right_symbol.len = 0;  // Mark as deleted
        
        // Update linked list
        left_symbol.next = right_symbol.next;
        
        // Add new potential merges
        add_new_bigram(left_symbol.prev, left);
        add_new_bigram(left, left_symbol.next);
    }
}
```

**Key insight**: Uses **priority queue** not linear merging. Rank from merge list = priority.

### 3. Text Decoding (No Ġ prefix!)

```cpp
// From llama.cpp token_to_piece for BPE:
std::string result = llama_decode_text(token_text);

// llama_decode_text converts byte sequences:
for (codepoint in unicode_cpts_from_utf8(text)) {
    result += unicode_utf8_to_byte(codepoint);
}
```

**My error in analysis**: I said GPT-2 uses `Ġ` prefix. Actually:
- GPT-2 tokens are **byte-level encoded**
- Each byte 0x00-0xFF maps to a Unicode codepoint
- Decoding converts Unicode back to bytes

**Our approach**: Copy the byte-to-unicode mapping from llama.cpp.

---

## Rust Implementation Architecture

### Dependencies Needed

```toml
[dependencies]
regex = "1.10"  # For pre-tokenization patterns
unicode-segmentation = "1.11"  # For UTF-8 character splitting
```

### Data Structures

```rust
/// BPE merge with priority rank
struct BPEMerge {
    left: String,
    right: String,
    rank: usize,  // Position in merge list = priority
}

/// Symbol in the working token list
struct Symbol {
    text_start: usize,  // Byte offset in original string
    text_len: usize,    // Byte length
    prev: Option<usize>,
    next: Option<usize>,
}

/// Bigram candidate for merging
#[derive(Eq, PartialEq)]
struct Bigram {
    left: usize,
    right: usize,
    rank: usize,  // Lower = higher priority
    text: String,
}

impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order: lower rank = higher priority
        other.rank.cmp(&self.rank)
            .then_with(|| other.left.cmp(&self.left))
    }
}
```

### Algorithm Steps

```rust
impl BPETokenizer {
    pub fn encode(&self, text: &str, vocab: &Vocabulary) -> Vec<TokenId> {
        // Step 1: Pre-tokenization via regex
        let fragments = self.pre_tokenize(text, vocab.pre_type());
        
        // Step 2: For each fragment, apply BPE
        let mut result = Vec::new();
        for fragment in fragments {
            let tokens = self.bpe_fragment(&fragment, vocab);
            result.extend(tokens);
        }
        
        result
    }
    
    fn pre_tokenize(&self, text: &str, pre_type: &str) -> Vec<String> {
        let pattern = match pre_type {
            "gpt2" | "gpt-2" => GPT2_PATTERN,
            "llama3" | "llama-bpe" => LLAMA3_PATTERN,
            _ => GPT2_PATTERN,  // Default
        };
        
        let re = Regex::new(pattern).unwrap();
        re.find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect()
    }
    
    fn bpe_fragment(&self, text: &str, vocab: &Vocabulary) -> Vec<TokenId> {
        // Build merge rank map
        let merge_ranks: HashMap<(String, String), usize> = 
            vocab.get_merges()
                .iter()
                .enumerate()
                .map(|(rank, (l, r))| ((l.clone(), r.clone()), rank))
                .collect();
        
        // Split into UTF-8 characters
        let mut symbols = Vec::new();
        let mut offset = 0;
        for (i, ch) in text.char_indices() {
            let next_offset = text.char_indices()
                .nth(i + 1)
                .map(|(pos, _)| pos)
                .unwrap_or(text.len());
            
            symbols.push(Symbol {
                text_start: offset,
                text_len: next_offset - offset,
                prev: if i == 0 { None } else { Some(i - 1) },
                next: if next_offset == text.len() { None } else { Some(i + 1) },
            });
            offset = next_offset;
        }
        
        // Build initial work queue
        let mut work_queue = BinaryHeap::new();
        for i in 0..symbols.len().saturating_sub(1) {
            self.try_add_bigram(i, i + 1, text, &symbols, &merge_ranks, &mut work_queue);
        }
        
        // Apply merges
        while let Some(bigram) = work_queue.pop() {
            // Validate bigram is still valid
            if symbols[bigram.left].text_len == 0 || 
               symbols[bigram.right].text_len == 0 ||
               symbols[bigram.left].next != Some(bigram.right) {
                continue;
            }
            
            // Merge
            symbols[bigram.left].text_len += symbols[bigram.right].text_len;
            symbols[bigram.right].text_len = 0;
            
            // Update linked list
            symbols[bigram.left].next = symbols[bigram.right].next;
            if let Some(next) = symbols[bigram.right].next {
                symbols[next].prev = Some(bigram.left);
            }
            
            // Add new potential merges
            if let Some(prev) = symbols[bigram.left].prev {
                self.try_add_bigram(prev, bigram.left, text, &symbols, &merge_ranks, &mut work_queue);
            }
            if let Some(next) = symbols[bigram.left].next {
                self.try_add_bigram(bigram.left, next, text, &symbols, &merge_ranks, &mut work_queue);
            }
        }
        
        // Convert symbols to token IDs
        let mut result = Vec::new();
        for sym in &symbols {
            if sym.text_len > 0 {
                let token_text = &text[sym.text_start..sym.text_start + sym.text_len];
                if let Some(id) = vocab.get_token_id(token_text) {
                    result.push(id);
                } else {
                    // Byte fallback
                    for byte in token_text.bytes() {
                        result.push(vocab.byte_to_token(byte));
                    }
                }
            }
        }
        
        result
    }
    
    fn try_add_bigram(
        &self,
        left: usize,
        right: usize,
        text: &str,
        symbols: &[Symbol],
        merge_ranks: &HashMap<(String, String), usize>,
        work_queue: &mut BinaryHeap<Bigram>,
    ) {
        let left_text = &text[symbols[left].text_start..symbols[left].text_start + symbols[left].text_len];
        let right_text = &text[symbols[right].text_start..symbols[right].text_start + symbols[right].text_len];
        
        if let Some(&rank) = merge_ranks.get(&(left_text.to_string(), right_text.to_string())) {
            work_queue.push(Bigram {
                left,
                right,
                rank,
                text: format!("{}{}", left_text, right_text),
            });
        }
    }
}
```

---

## Implementation Tasks (Fibonacci 8 = ~6 hours)

### Task 1: Add Dependencies (Fibonacci 1 = 15 min)
- Add `regex` and `unicode-segmentation` to Cargo.toml
- Test compilation

### Task 2: Pre-Tokenization Patterns (Fibonacci 2 = 30 min)
- Define GPT-2 pattern constant
- Define Llama-3 pattern constant
- Implement `pre_tokenize()` method
- Test pattern matching

### Task 3: Core BPE Algorithm (Fibonacci 5 = 3 hours)
- Implement Symbol struct
- Implement Bigram struct with Ord
- Implement `bpe_fragment()` with priority queue
- Implement `try_add_bigram()` helper
- Test merge algorithm

### Task 4: Decode Implementation (Fibonacci 2 = 30 min)
- Keep current decode (just concatenate token text)
- llama.cpp handles byte decoding in token_to_piece
- We already have byte fallback

### Task 5: Testing & Validation (Fibonacci 3 = 1.5 hours)
- Create test with GPT-2 GGUF file (if available)
- Compare with llama.cpp output
- Debug mismatches
- Document any limitations

---

## Total Estimate: Fibonacci 13 = ~6 hours

**Breakdown**:
- 15 min: Dependencies
- 30 min: Regex patterns  
- 3 hours: Core algorithm
- 30 min: Decode
- 1.5 hours: Testing
- **Total: 6 hours**

---

## Success Criteria

1. ✅ Can tokenize GPT-2 style text
2. ✅ Priority-based merge algorithm works
3. ✅ Matches llama.cpp output (if GPT-2 model available)
4. ✅ No panics or errors
5. ✅ Clean compilation

---

## Notes

- **No `Ġ` prefix needed** - that was my misunderstanding
- **Byte-level encoding** handled by vocab lookup, not BPE algorithm
- **Regex crate** well-established in Rust, no concerns
- **Priority queue** = BinaryHeap (std lib), perfect match

---

## Start Implementation NOW

Ready to implement. Estimated completion: Today (6 hours of work).
