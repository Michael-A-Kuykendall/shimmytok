//! BPE (Byte Pair Encoding) tokenizer implementation.
//!
//! This module implements GPT-2 style Byte Pair Encoding with direct merge rules from GGUF files.
//! It supports both single-pattern models (GPT-2, Llama-3) and multi-pattern sequential
//! tokenization (DeepSeek, StarCoder).
//!
//! # Architecture
//! - **Pre-tokenization**: Regex-based text splitting (41 model-specific patterns from llama.cpp)
//! - **BPE Merging**: Priority queue-based symbol merging using vocabulary merge rules
//! - **Multi-pattern**: Sequential pattern application with gap preservation
//!
//! # Reference Implementation
//! Direct port of llama.cpp's tokenizer:
//! - Pattern definitions: `llama-vocab.cpp` lines 300–450
//! - Sequential splitting: `unicode.cpp` `unicode_regex_split_stl()`
//! - BPE merging: `llm_tokenizer_bpe_session::tokenize()` lines 1040–1118
//!
//! # Model Coverage
//! Supports ~95 % of popular model families including GPT-2, Llama-3, Qwen2,
//! DeepSeek (LLM/Coder/V3/R1), StarCoder, Phi-2, Mistral, ChatGLM4, DBRX,
//! Falcon, Bloom, Tekken, Chameleon, GPT-4o, and Grok-2.

// Model names like "GPT-2", "DeepSeek" etc. inside doc comments are not Rust items.
#![allow(clippy::doc_markdown)]

use crate::vocab::Vocabulary;
use crate::TokenId;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// Symbol representing a text fragment during BPE merging
#[derive(Debug, Clone)]
struct Symbol {
    /// Start byte offset in original text
    text_start: usize,
    /// Byte length of this symbol
    text_len: usize,
    /// Index of previous symbol (linked list)
    prev: Option<usize>,
    /// Index of next symbol (linked list)
    next: Option<usize>,
}

/// Bigram candidate for merging, with priority rank.
///
/// Note: the `text` field is populated but intentionally unused after construction —
/// it exists to match llama.cpp's data structure layout and may be used in future
/// debug tooling.
#[derive(Debug, Clone, Eq, PartialEq)]
struct Bigram {
    left: usize,
    right: usize,
    rank: usize,
}

impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> Ordering {
        // Lower rank = higher priority, so reverse the comparison
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Immutable, model-specific BPE state prepared once at construction and reused
/// by every `encode` call.
///
/// Moving this work out of the hot path removes the per-call mutex lock, regex
/// clone, and `String`-keyed merge-rank rebuild that the previous implementation
/// performed on every encode.
struct BPEPreparedState {
    /// Pre-tokenization regexes for this model, compiled once and applied
    /// sequentially. No mutex, no per-call clone.
    regexes: Vec<fancy_regex::Regex>,
    /// Merge priorities keyed by `(left_token_id, right_token_id)`. Built once
    /// from the vocabulary; lookups need no `String` allocation.
    merge_ranks: HashMap<(TokenId, TokenId), usize>,
    /// llama3 `ignore_merges` optimization flag, derived from the pre-type.
    ignore_merges: bool,
}

pub struct BPETokenizer {
    prepared: BPEPreparedState,
}

impl BPETokenizer {
    /// Build a BPE tokenizer with all model-specific state prepared up front.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::TokenizationFailed`] if a pre-tokenization regex
    /// for the model's pre-type fails to compile.
    pub fn new(vocab: &Vocabulary) -> Result<Self, crate::Error> {
        let pre_type = vocab.pre_type().unwrap_or("default");

        // Compile the pre-tokenization regexes once.
        let patterns = Self::get_patterns(pre_type);
        let mut regexes = Vec::with_capacity(patterns.len());
        for pattern in patterns {
            let regex = fancy_regex::Regex::new(pattern).map_err(|e| {
                crate::Error::TokenizationFailed(format!(
                    "Failed to compile regex for '{pre_type}': {e}"
                ))
            })?;
            regexes.push(regex);
        }

        // Build merge ranks keyed by token IDs. The vocabulary validates at load
        // time that every merge rule references known tokens, so `get_token_id`
        // is expected to succeed; unknown pairs are skipped defensively.
        let merge_ranks: HashMap<(TokenId, TokenId), usize> = vocab
            .get_merges()
            .iter()
            .enumerate()
            .filter_map(
                |(rank, (l, r))| match (vocab.get_token_id(l), vocab.get_token_id(r)) {
                    (Some(lid), Some(rid)) => Some(((lid, rid), rank)),
                    _ => None,
                },
            )
            .collect();

        let ignore_merges = matches!(vocab.pre_type(), Some(p) if
            matches!(p, "llama3" | "llama-v3" | "llama-bpe"));

        Ok(Self {
            prepared: BPEPreparedState {
                regexes,
                merge_ranks,
                ignore_merges,
            },
        })
    }

    /// Get pre-tokenization regex patterns for a given model type.
    ///
    /// Returns patterns that are applied **sequentially** (not as alternates in a single regex).
    /// This matches llama.cpp's implementation where each pattern refines the tokenization boundaries.
    ///
    /// # Multi-Pattern Strategy
    /// For models with multiple patterns (e.g., DeepSeek-LLM with 6 patterns):
    /// 1. Start with full text as single fragment
    /// 2. Apply pattern 1: split matches vs non-matches (preserve both)
    /// 3. Apply pattern 2 to each fragment from step 2, further refining
    /// 4. Continue until all patterns applied
    ///
    /// This differs from single-pattern models (GPT-2, Llama-3) that match directly.
    ///
    /// # Reference
    /// Based on llama.cpp `llama-vocab.cpp` lines 300-450 (model-specific pattern definitions)
    /// and `unicode.cpp` `unicode_regex_split_stl()` for sequential application logic.
    fn get_patterns(pre_type: &str) -> Vec<&'static str> {
        match pre_type {
            // Llama-3 family
            "llama3" | "llama-v3" | "llama-bpe" => vec![
                r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            ],

            // DeepSeek family (patterns from llama.cpp llama-vocab.cpp lines 323-332)
            // Note: Original deepseek-llm pattern has explicit Unicode ranges that include astral plane
            // characters (e.g., 𐐀-𐑏) which Rust's regex crate doesn't fully support.
            // We use \p{L} as a practical approximation that covers most use cases.
            "deepseek-llm" => vec![
                r"[\r\n]",
                r"\s?\p{L}+", // Simplified from explicit Unicode ranges
                r"\s?[!-/:-~！-／：-～'-‟　-。]+",
                r"\s+$",
                r"[一-龥ࠀ-一가-퟿]+",
                r"\p{N}+",
            ],
            "deepseek-coder" => vec![
                r"[\r\n]",
                r"\s?\p{L}+",
                r"\s?\p{P}+",
                r"[一-龥ࠀ-一가-퟿]+",
                r"\p{N}",
            ],
            "deepseek-v3" => vec![
                r"\p{N}{1,3}",
                r"[一-龥぀-ゟ゠-ヿ]+",
                r"[!#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            ],
            "deepseek-r1-qwen" => vec![
                r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            ],

            // Falcon
            "falcon" => vec![r"\n| ?[\p{L}\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+"],

            // StarCoder family (TWO patterns!)
            "starcoder" | "refact" | "command-r" | "smollm" | "codeshell" | "exaone"
            | "minerva" => vec![
                r"\p{N}", // First: split individual digits
                r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)",
            ],

            // GPT-2 family
            "gpt-2" | "phi-2" | "jina-es" | "jina-de" | "mpt" | "olmo" | "jais" | "trillion"
            | "granite-docling" | "exaone4" => {
                vec![r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"]
            }

            // Qwen2 family
            "qwen2" | "stablelm2" | "hunyuan" | "megrez" => vec![
                r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            ],

            // Bloom family
            "bloom" | "poro-chat" | "gpt3-finnish" => vec![r"\s+|\S+"],

            // ChatGLM — same pattern as Llama-3 but with 1–3 digit numeric chunks
            "chatglm4" | "glm4" | "chatglm-bpe" | "dbrx" | "smaug-bpe" => vec![
                r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            ],

            // Norwegian
            "viking" => vec![r" ?[^(\s|.,!?…。，、।۔،)]+"],

            // Advanced/Specialized
            "tekken" => vec![
                r"[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))*((?=[\p{L}])([^A-Z]))+|[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))+((?=[\p{L}])([^A-Z]))*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            ],
            "chameleon" => vec![
                r"<sentinel:[0-9]+>|(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z|([\t\n]|    |  )|\p{N}|[\p{P}!-/:-@\[-`{-~]|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)",
            ],
            "gpt-4o" | "llama4" => vec![
                r"[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))*((?=[\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))+((?=[\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            ],
            "kimi-k2" => vec![r"\p{Han}+"],
            "superbpe" => vec![r"\p{N}+|(?=(\d{3})+(?!\d))"],
            "bailingmoe" | "bailingmoe2" | "llada-moe" => vec![
                r"'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+",
            ],
            "seed-coder" => vec![
                r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1}| ?[^\s\p{L}\p{N}\r\n]+|\s*[\r\n]+|\s+(?!\S)|\s+",
            ],
            "hunyuan-dense" => vec![
                r"\p{N}{1,3}",
                r"[一-龥぀-ゟ゠-ヿ]+",
                r"[!#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            ],
            "grok-2" => vec![
                r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            ],

            // Default case (from llama.cpp lines 439-446)
            // Used when model file doesn't specify pre-tokenizer type.
            // Four patterns matching llama.cpp exactly.
            _ => vec![
                r"[\p{P}\$\+<=>^~\|]+",
                r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)",
                r"\p{N}+",
                r"[0-9][0-9][0-9]",
            ],
        }
    }

    /// Pre-tokenize text into fragments using sequential regex pattern matching.
    ///
    /// # Algorithm
    /// Implements llama.cpp's offset-based approach (`unicode_regex_split_stl`):
    /// - Single pattern: Direct regex matching (fast path)
    /// - Multiple patterns: Sequential refinement preserving both matches AND gaps
    ///
    /// ## Multi-Pattern Example
    /// Text: "Hello123World"
    /// Pattern 1: `\p{N}+` → matches "123"
    /// Result: ["Hello", "123", "World"] (gaps "Hello"/"World" preserved)
    ///
    /// Pattern 2: `\p{L}+` → matches letters in each fragment
    /// Result: ["Hello", "123", "World"] ("123" has no letters, passes through)
    ///
    /// # Gap Preservation
    /// CRITICAL: Non-matching regions between matches are preserved as separate fragments.
    /// This prevents information loss and matches llama.cpp's behavior.
    ///
    /// # Reference
    /// llama.cpp `unicode.cpp` lines 531-563 (`unicode_regex_split_stl`)
    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        let regexes = &self.prepared.regexes;

        if regexes.len() == 1 {
            // Fast path for single-pattern models (most common)
            return regexes[0]
                .find_iter(text)
                .filter_map(std::result::Result::ok)
                .map(|m| m.as_str().to_string())
                .collect();
        }

        // For multiple patterns, use offset-based approach like llama.cpp
        // Each pattern refines the boundaries, preserving both matches and non-matches
        let mut offsets: Vec<(usize, usize)> = vec![(0, text.len())];

        for regex in regexes {
            let mut new_offsets = Vec::new();

            for (start, end) in offsets {
                let fragment = &text[start..end];

                // Collect all matches in this fragment
                let matches: Vec<_> = regex
                    .find_iter(fragment)
                    .filter_map(std::result::Result::ok)
                    .collect();

                if matches.is_empty() {
                    // No matches - keep the original offset unchanged
                    new_offsets.push((start, end));
                } else {
                    // Split into matched and unmatched regions
                    let mut last_pos = 0;

                    for m in matches {
                        // Add unmatched gap before this match
                        if m.start() > last_pos {
                            new_offsets.push((start + last_pos, start + m.start()));
                        }

                        // Add the match
                        new_offsets.push((start + m.start(), start + m.end()));
                        last_pos = m.end();
                    }

                    // Add final unmatched portion
                    if last_pos < fragment.len() {
                        new_offsets.push((start + last_pos, end));
                    }
                }
            }

            offsets = new_offsets;
        }

        // Convert offsets to strings
        offsets
            .iter()
            .map(|(start, end)| text[*start..*end].to_string())
            .collect()
    }

    /// Apply Byte Pair Encoding merge algorithm to a single text fragment.
    ///
    /// # Algorithm Overview (from llama.cpp)
    /// 1. **Initialize symbols**: Split text into UTF-8 characters as initial symbols
    /// 2. **Build merge rank map**: Create (left, right) → priority mapping from vocab
    /// 3. **Create work queue**: Priority queue of all adjacent bigram candidates
    /// 4. **Merge loop**: Pop highest-priority bigram, validate, merge, add new neighbors
    /// 5. **Convert to tokens**: Map final symbols to token IDs (with byte fallback)
    ///
    /// # Data Structures
    /// - `Symbol`: Text fragment with byte position + doubly-linked list pointers
    /// - `Bigram`: Merge candidate with (left_idx, right_idx, rank, text)
    /// - `BinaryHeap`: Priority queue ordered by merge rank (lower rank = higher priority)
    ///
    /// # Merge Validation
    /// CRITICAL: Before merging, validates that symbol texts still match the merge rule.
    /// Symbols may have changed since bigram was added to queue (due to earlier merges).
    ///
    /// # Reference
    /// Direct port of llama.cpp `llm_tokenizer_bpe_session::tokenize` (lines 1040-1118)
    fn bpe_fragment(&self, text: &str, vocab: &Vocabulary) -> Result<Vec<TokenId>, crate::Error> {
        let merge_ranks = &self.prepared.merge_ranks;
        // Split into UTF-8 characters as initial symbols
        let char_indices: Vec<(usize, char)> = text.char_indices().collect();
        let mut symbols: Vec<Symbol> = Vec::with_capacity(char_indices.len());

        for (i, (byte_pos, _ch)) in char_indices.iter().enumerate() {
            let next_byte_pos = if i + 1 < char_indices.len() {
                char_indices[i + 1].0
            } else {
                text.len()
            };

            symbols.push(Symbol {
                text_start: *byte_pos,
                text_len: next_byte_pos - byte_pos,
                prev: if i == 0 { None } else { Some(i - 1) },
                next: if i + 1 < char_indices.len() {
                    Some(i + 1)
                } else {
                    None
                },
            });
        }

        if symbols.is_empty() {
            return Ok(Vec::new());
        }

        // Build initial work queue with all adjacent bigrams
        let mut work_queue = BinaryHeap::new();
        for i in 0..symbols.len().saturating_sub(1) {
            if let Some(next) = symbols[i].next {
                try_add_bigram(i, next, text, &symbols, merge_ranks, vocab, &mut work_queue);
            }
        }

        // Apply merges in priority order
        while let Some(bigram) = work_queue.pop() {
            let left = bigram.left;
            let right = bigram.right;

            // Validate bigram is still valid
            if symbols[left].text_len == 0
                || symbols[right].text_len == 0
                || symbols[left].next != Some(right)
            {
                continue;
            }

            // CRITICAL: Validate that current symbol texts match the merge rule
            // Symbols may have changed since bigram was added to queue
            let left_text =
                &text[symbols[left].text_start..symbols[left].text_start + symbols[left].text_len];
            let right_text = &text
                [symbols[right].text_start..symbols[right].text_start + symbols[right].text_len];

            // Look up token IDs to key the merge rule without allocating Strings.
            // A pair can only be a merge rule if both sides are vocabulary tokens.
            let rank = match (
                vocab.get_token_id(left_text),
                vocab.get_token_id(right_text),
            ) {
                (Some(lid), Some(rid)) => merge_ranks.get(&(lid, rid)).copied(),
                _ => None,
            };

            if let Some(expected_rank) = rank {
                if expected_rank == bigram.rank {
                    // Merge: extend left symbol to include right symbol
                    symbols[left].text_len += symbols[right].text_len;
                    symbols[right].text_len = 0; // Mark right as deleted

                    // Update linked list
                    symbols[left].next = symbols[right].next;
                    if let Some(next) = symbols[right].next {
                        symbols[next].prev = Some(left);
                    }

                    // Add new potential merges with neighbors
                    if let Some(prev) = symbols[left].prev {
                        try_add_bigram(
                            prev,
                            left,
                            text,
                            &symbols,
                            merge_ranks,
                            vocab,
                            &mut work_queue,
                        );
                    }
                    if let Some(next) = symbols[left].next {
                        try_add_bigram(
                            left,
                            next,
                            text,
                            &symbols,
                            merge_ranks,
                            vocab,
                            &mut work_queue,
                        );
                    }
                }
            }
        }

        // Convert symbols to token IDs (llama.cpp lines 1101-1118)
        let mut result = Vec::new();
        for sym in &symbols {
            if sym.text_len > 0 {
                let token_text = &text[sym.text_start..sym.text_start + sym.text_len];
                if let Some(id) = vocab.get_token_id(token_text) {
                    result.push(id);
                } else {
                    // Byte fallback: look up each byte-encoded character individually.
                    // NOT using hex format <0xXX> — that's SentencePiece only.
                    for byte_char in token_text.chars() {
                        let byte_str = byte_char.to_string();
                        if let Some(id) = vocab.get_token_id(&byte_str) {
                            result.push(id);
                        } else {
                            result.push(vocab.unk_token_id());
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    /// Encode text to token IDs using BPE.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::TokenizationFailed`] if:
    /// - `text` exceeds [`crate::MAX_INPUT_SIZE`]
    /// - the pre-tokenization regex fails to compile or match
    /// - the encoded output would exceed [`crate::MAX_OUTPUT_TOKENS`]
    pub fn encode(&self, text: &str, vocab: &Vocabulary) -> Result<Vec<TokenId>, crate::Error> {
        // Validate input size
        if text.len() > crate::MAX_INPUT_SIZE {
            return Err(crate::Error::TokenizationFailed(format!(
                "Input text too large: {} bytes (max: {})",
                text.len(),
                crate::MAX_INPUT_SIZE
            )));
        }

        // Pre-tokenize the original text (not byte-encoded) into word fragments.
        // Regexes were compiled once at construction — no lock, no clone here.
        let fragments = self.pre_tokenize(text);

        // Apply BPE to each fragment (after GPT-2 byte-encoding). Merge ranks
        // and the ignore-merges flag were prepared once at construction.
        let mut result = Vec::new();
        let ignore_merges = self.prepared.ignore_merges;
        for fragment in fragments {
            let fragment_encoded = crate::byte_encoder::encode_bytes(&fragment);
            // llama.cpp `tokenizer_ignore_merges` optimization: if the whole
            // byte-encoded fragment is already a single vocabulary token, emit
            // it directly without running the merge algorithm. This is required
            // for llama3/llama-bpe models where many multi-char tokens exist
            // that have no explicit merge rules (e.g. "Ġ{" = token 314).
            if ignore_merges {
                if let Some(tok_id) = vocab.get_token_id(&fragment_encoded) {
                    result.push(tok_id);
                    continue;
                }
            }
            let tokens = self.bpe_fragment(&fragment_encoded, vocab)?;
            if result.len() + tokens.len() > crate::MAX_OUTPUT_TOKENS {
                return Err(crate::Error::TokenizationFailed(format!(
                    "Output would exceed max tokens: {} (max: {})",
                    result.len() + tokens.len(),
                    crate::MAX_OUTPUT_TOKENS
                )));
            }
            result.extend(tokens);
        }

        Ok(result)
    }

    /// Decode token IDs back to text.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::InvalidToken`] if any token ID is not present in the
    /// vocabulary, or [`crate::Error::TokenizationFailed`] if the decoded output
    /// would exceed 100 MB.
    pub fn decode(&self, tokens: &[TokenId], vocab: &Vocabulary) -> Result<String, crate::Error> {
        // Validate all token IDs exist
        for &id in tokens {
            if vocab.get_token_text(id).is_none() {
                return Err(crate::Error::InvalidToken(format!(
                    "Token ID {id} not found in vocabulary"
                )));
            }
        }

        let byte_encoded_text: String = tokens
            .iter()
            .filter_map(|&id| vocab.get_token_text(id))
            .collect::<Vec<_>>()
            .join("");

        // Convert from GPT-2 byte encoding back to normal UTF-8
        let decoded = crate::byte_encoder::decode_bytes(&byte_encoded_text);

        // Validate final decoded size (Issue R3#8) - decoding can expand
        const MAX_DECODED_SIZE: usize = 100 * 1024 * 1024; // 100MB
        if decoded.len() > MAX_DECODED_SIZE {
            return Err(crate::Error::TokenizationFailed(format!(
                "Final decoded text too large: {} bytes (max: {})",
                decoded.len(),
                MAX_DECODED_SIZE
            )));
        }

        Ok(decoded)
    }
}

/// Try to add a bigram to the work queue if the two symbols form a valid merge pair.
///
/// Extracts the text of both symbols from `text` and looks up the merge rank.
/// Does nothing if either symbol has zero length (already merged).
fn try_add_bigram(
    left: usize,
    right: usize,
    text: &str,
    symbols: &[Symbol],
    merge_ranks: &HashMap<(TokenId, TokenId), usize>,
    vocab: &Vocabulary,
    work_queue: &mut BinaryHeap<Bigram>,
) {
    if symbols[left].text_len == 0 || symbols[right].text_len == 0 {
        return;
    }

    let left_text =
        &text[symbols[left].text_start..symbols[left].text_start + symbols[left].text_len];
    let right_text =
        &text[symbols[right].text_start..symbols[right].text_start + symbols[right].text_len];

    // Key the merge lookup by token IDs — no String allocation per candidate.
    // If either side is not a vocabulary token, the pair cannot be a merge rule.
    if let (Some(left_id), Some(right_id)) = (
        vocab.get_token_id(left_text),
        vocab.get_token_id(right_text),
    ) {
        if let Some(&rank) = merge_ranks.get(&(left_id, right_id)) {
            work_queue.push(Bigram { left, right, rank });
        }
    }
}

impl crate::TokenizerImpl for BPETokenizer {
    fn encode(&self, text: &str, vocab: &Vocabulary) -> Result<Vec<TokenId>, crate::Error> {
        BPETokenizer::encode(self, text, vocab)
    }

    fn decode(&self, tokens: &[TokenId], vocab: &Vocabulary) -> Result<String, crate::Error> {
        BPETokenizer::decode(self, tokens, vocab)
    }
}
