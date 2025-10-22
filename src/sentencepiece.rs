//! SentencePiece tokenizer implementation
//! Direct port of llama.cpp's algorithm including resegment

use crate::{TokenId, TokenizerImpl, Vocabulary};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// Symbol represents a UTF-8 character or merged sequence
#[derive(Debug, Clone)]
struct Symbol {
    // Instead of storing text, store position and length in original string
    pos: usize,          // Position in processed_text
    len: usize,          // Length in bytes
    prev: Option<usize>, // Previous symbol in linked list
    next: Option<usize>, // Next symbol in linked list
}

/// Bigram candidate for merging
#[derive(Debug, Clone)]
struct Bigram {
    left: usize,  // Left symbol index
    right: usize, // Right symbol index
    score: f32,   // Token score from vocabulary
    size: usize,  // Combined byte size
}

impl PartialEq for Bigram {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.left == other.left
    }
}

impl Eq for Bigram {}

impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher score wins (less negative is better)
        // Handle NaN by treating as lowest priority
        match self.score.partial_cmp(&other.score) {
            Some(ord) => ord.then_with(|| other.left.cmp(&self.left)),
            None => {
                // If either score is NaN, fall back to position comparison
                // NaN scores are treated as lower priority than any real score
                if self.score.is_nan() && other.score.is_nan() {
                    other.left.cmp(&self.left)
                } else if self.score.is_nan() {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            }
        }
    }
}

pub struct SentencePieceTokenizer;

impl Default for SentencePieceTokenizer {
    fn default() -> Self {
        Self
    }
}

impl SentencePieceTokenizer {
    pub fn new() -> Self {
        Self
    }
}

impl TokenizerImpl for SentencePieceTokenizer {
    fn encode(&self, text: &str, vocab: &Vocabulary) -> Result<Vec<TokenId>, crate::Error> {
        // Validate input size
        const MAX_INPUT_SIZE: usize = 10 * 1024 * 1024; // 10MB
        if text.len() > MAX_INPUT_SIZE {
            return Err(crate::Error::TokenizationFailed(format!(
                "Input text too large: {} bytes (max: {})",
                text.len(),
                MAX_INPUT_SIZE
            )));
        }

        if text.is_empty() {
            return Ok(Vec::new());
        }

        // Add space prefix for SentencePiece (replacing spaces with ▁)
        let processed_text = if !text.starts_with(' ') {
            format!("▁{}", text.replace(' ', "▁"))
        } else {
            text.replace(' ', "▁")
        };

        // Split text into UTF-8 characters
        let mut symbols = Vec::new();
        let mut char_indices = processed_text.char_indices().peekable();
        let mut index = 0;

        while let Some((byte_pos, _ch)) = char_indices.next() {
            let next_pos = char_indices
                .peek()
                .map(|(pos, _)| *pos)
                .unwrap_or(processed_text.len());
            let len = next_pos - byte_pos;

            let prev = if index == 0 { None } else { Some(index - 1) };
            let next = if char_indices.peek().is_some() {
                Some(index + 1)
            } else {
                None
            };

            symbols.push(Symbol {
                pos: byte_pos,
                len,
                prev,
                next,
            });

            index += 1;
        }

        // Track merge history for resegment
        let mut rev_merge: HashMap<String, (usize, usize)> = HashMap::new();

        // Initialize work queue with all adjacent pairs
        let mut work_queue = BinaryHeap::new();
        for i in 1..symbols.len() {
            try_add_bigram(
                &processed_text,
                &symbols,
                i - 1,
                i,
                vocab,
                &mut work_queue,
                &mut rev_merge,
            );
        }

        // Process merges in priority order
        // Add iteration limit to prevent infinite loops
        let max_iterations = 10 * symbols.len().max(1);
        let mut iterations = 0;
        while let Some(bigram) = work_queue.pop() {
            iterations += 1;
            if iterations > max_iterations {
                return Err(crate::Error::TokenizationFailed(
                    "SentencePiece merge iteration limit exceeded".to_string()
                ));
            }
            
            if bigram.left >= symbols.len() || bigram.right >= symbols.len() {
                continue;
            }

            let left_sym = &symbols[bigram.left];
            let right_sym = &symbols[bigram.right];

            // Skip if already merged (len = 0 means merged)
            if left_sym.len == 0 || right_sym.len == 0 {
                continue;
            }

            // Skip if symbols are no longer adjacent or size changed
            if left_sym.next != Some(bigram.right) || left_sym.len + right_sym.len != bigram.size {
                continue;
            }

            // Merge right into left (extend length, don't modify text)
            symbols[bigram.left].len += symbols[bigram.right].len;
            symbols[bigram.right].len = 0; // Mark as merged

            // Update linked list
            symbols[bigram.left].next = symbols[bigram.right].next;
            if let Some(next_idx) = symbols[bigram.right].next {
                symbols[next_idx].prev = Some(bigram.left);
            }

            // Try new bigrams with neighbors
            if let Some(prev) = symbols[bigram.left].prev {
                try_add_bigram(
                    &processed_text,
                    &symbols,
                    prev,
                    bigram.left,
                    vocab,
                    &mut work_queue,
                    &mut rev_merge,
                );
            }
            if let Some(next) = symbols[bigram.left].next {
                try_add_bigram(
                    &processed_text,
                    &symbols,
                    bigram.left,
                    next,
                    vocab,
                    &mut work_queue,
                    &mut rev_merge,
                );
            }
        }

        // Collect final tokens with resegment
        let mut result = Vec::new();
        let mut current = 0;

        // Find first symbol (the one with no prev)
        for (i, sym) in symbols.iter().enumerate() {
            if sym.prev.is_none() && sym.len > 0 {
                current = i;
                break;
            }
        }

        // Walk the linked list and resegment
        loop {
            if current >= symbols.len() {
                break;
            }

            let symbol = &symbols[current];
            if symbol.len > 0 {
                let text = &processed_text[symbol.pos..symbol.pos + symbol.len];
                resegment(
                    text,
                    &processed_text,
                    &symbols,
                    &rev_merge,
                    vocab,
                    &mut result,
                );
            }

            if let Some(next) = symbol.next {
                current = next;
            } else {
                break;
            }
        }

        Ok(result)
    }

    fn decode(&self, tokens: &[TokenId], vocab: &Vocabulary) -> Result<String, crate::Error> {
        // Validate all tokens exist
        for &token_id in tokens {
            if vocab.get_token_text(token_id).is_none() {
                return Err(crate::Error::InvalidToken(format!(
                    "Token ID {} not found in vocabulary",
                    token_id
                )));
            }
        }

        let mut result = String::new();

        for &token_id in tokens {
            if let Some(text) = vocab.get_token_text(token_id) {
                result.push_str(text);
            }
        }

        // Replace sentencepiece meta-symbol
        Ok(result.replace('▁', " "))
    }
}

fn try_add_bigram(
    text: &str,
    symbols: &[Symbol],
    left: usize,
    right: usize,
    vocab: &Vocabulary,
    work_queue: &mut BinaryHeap<Bigram>,
    rev_merge: &mut HashMap<String, (usize, usize)>,
) {
    if left >= symbols.len() || right >= symbols.len() {
        return;
    }

    let left_sym = &symbols[left];
    let right_sym = &symbols[right];

    if left_sym.len == 0 || right_sym.len == 0 {
        return;
    }

    // Get the combined text
    let combined_text = &text[left_sym.pos..left_sym.pos + left_sym.len + right_sym.len];

    // Check if this combination exists in vocabulary
    if let Some(token_id) = vocab.get_token_id(combined_text) {
        let score = vocab.get_token_score(token_id);

        work_queue.push(Bigram {
            left,
            right,
            score,
            size: left_sym.len + right_sym.len,
        });

        // Track merge history
        rev_merge.insert(combined_text.to_string(), (left, right));
    }
}

/// Resegment function - the critical missing piece from llama.cpp
fn resegment(
    text: &str,
    full_text: &str,
    symbols: &[Symbol],
    rev_merge: &HashMap<String, (usize, usize)>,
    vocab: &Vocabulary,
    output: &mut Vec<TokenId>,
) {
    // Try to find the text as a complete token
    if let Some(token_id) = vocab.get_token_id(text) {
        output.push(token_id);
        return;
    }

    // If not found, check if we have merge history for this text
    if let Some(&(left_idx, right_idx)) = rev_merge.get(text) {
        // Recursively resegment the parts
        if left_idx < symbols.len() && right_idx < symbols.len() {
            let left_sym = &symbols[left_idx];
            let right_sym = &symbols[right_idx];

            if left_sym.len > 0 {
                let left_text = &full_text[left_sym.pos..left_sym.pos + left_sym.len];
                resegment(left_text, full_text, symbols, rev_merge, vocab, output);
            }
            if right_sym.len > 0 {
                let right_text = &full_text[right_sym.pos..right_sym.pos + right_sym.len];
                resegment(right_text, full_text, symbols, rev_merge, vocab, output);
            }
            return;
        }
    }

    // Fallback: output as individual bytes
    for byte in text.bytes() {
        output.push(vocab.byte_to_token(byte));
    }
}
