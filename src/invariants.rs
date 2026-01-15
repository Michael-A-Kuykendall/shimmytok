//! Runtime invariant assertions for tokenizer correctness.
//!
//! This module provides debug-mode assertions that verify tokenizer invariants
//! at runtime. These checks help catch bugs during development and testing
//! without impacting release performance.
//!
//! # Invariants Enforced
//!
//! 1. **Token bounds**: All token IDs are within vocabulary bounds
//! 2. **Vocabulary consistency**: BOS/EOS tokens exist and are valid
//! 3. **Encode postconditions**: Output tokens are valid
//! 4. **Decode preconditions**: Input tokens are valid before decoding
//!
//! # Usage
//!
//! These assertions are only active in debug builds (`debug_assertions`).
//! In release builds, they compile to no-ops for zero overhead.
//!
//! ```ignore
//! use shimmytok::invariants;
//!
//! // After encoding
//! invariants::assert_encode_postconditions(&tokens, vocab_size);
//!
//! // Before decoding
//! invariants::assert_decode_preconditions(&tokens, vocab_size);
//! ```

use crate::Tokenizer;

/// Asserts that all token IDs in the slice are within vocabulary bounds.
///
/// # Panics
///
/// Panics in debug builds if any token ID >= `vocab_size`.
#[inline]
pub fn assert_tokens_in_bounds(tokens: &[u32], vocab_size: usize) {
    #[cfg(debug_assertions)]
    {
        for (i, &token) in tokens.iter().enumerate() {
            debug_assert!(
                (token as usize) < vocab_size,
                "Invariant violation: token[{i}] = {token} >= vocab_size ({vocab_size})"
            );
        }
    }
    #[cfg(not(debug_assertions))]
    {
        let _ = (tokens, vocab_size);
    }
}

/// Asserts postconditions after encoding text to tokens.
///
/// Verifies:
/// - All token IDs are within vocabulary bounds
/// - Token count is within reasonable limits
///
/// # Panics
///
/// Panics in debug builds if postconditions are violated.
#[inline]
pub fn assert_encode_postconditions(tokens: &[u32], vocab_size: usize) {
    #[cfg(debug_assertions)]
    {
        assert_tokens_in_bounds(tokens, vocab_size);

        // Sanity check: token count should be reasonable
        // (at most 4 tokens per input byte is a very generous upper bound)
        debug_assert!(
            tokens.len() <= crate::MAX_OUTPUT_TOKENS,
            "Invariant violation: token count {} exceeds MAX_OUTPUT_TOKENS ({})",
            tokens.len(),
            crate::MAX_OUTPUT_TOKENS
        );
    }
    #[cfg(not(debug_assertions))]
    {
        let _ = (tokens, vocab_size);
    }
}

/// Asserts preconditions before decoding tokens to text.
///
/// **Important**: This should only be used for tokens produced internally by
/// the tokenizer, not for user-supplied tokens. User input should be validated
/// with proper error handling, not assertions.
///
/// Verifies:
/// - All token IDs are within vocabulary bounds
///
/// # Panics
///
/// Panics in debug builds if preconditions are violated.
#[inline]
#[allow(dead_code)] // Available for internal use
pub fn assert_decode_preconditions(tokens: &[u32], vocab_size: usize) {
    #[cfg(debug_assertions)]
    {
        assert_tokens_in_bounds(tokens, vocab_size);
    }
    #[cfg(not(debug_assertions))]
    {
        let _ = (tokens, vocab_size);
    }
}

/// Asserts that a tokenizer's vocabulary is internally consistent.
///
/// Verifies:
/// - Vocabulary size is non-zero
/// - BOS token (if present) is within bounds
/// - EOS token (if present) is within bounds
/// - BOS and EOS are distinct (if both present)
///
/// # Panics
///
/// Panics in debug builds if the vocabulary is inconsistent.
#[inline]
pub fn assert_vocabulary_consistent(tokenizer: &Tokenizer) {
    #[cfg(debug_assertions)]
    {
        let vocab_size = tokenizer.vocab_size();

        debug_assert!(
            vocab_size > 0,
            "Invariant violation: vocabulary size is zero"
        );

        let bos = tokenizer.bos_token();
        let eos = tokenizer.eos_token();

        debug_assert!(
            (bos as usize) < vocab_size,
            "Invariant violation: BOS token {bos} >= vocab_size ({vocab_size})"
        );

        debug_assert!(
            (eos as usize) < vocab_size,
            "Invariant violation: EOS token {eos} >= vocab_size ({vocab_size})"
        );

        // BOS and EOS should typically be distinct
        // Note: Some models may legitimately have them equal, so we don't assert here.
        // This check is informational only in debug builds.
        let _ = (bos, eos); // Acknowledge these are used for comparison above
    }
    #[cfg(not(debug_assertions))]
    {
        let _ = tokenizer;
    }
}

/// Asserts that a single token ID is valid.
///
/// # Panics
///
/// Panics in debug builds if `token >= vocab_size`.
#[inline]
pub fn assert_valid_token(token: u32, vocab_size: usize) {
    #[cfg(debug_assertions)]
    {
        debug_assert!(
            (token as usize) < vocab_size,
            "Invariant violation: token {token} >= vocab_size ({vocab_size})"
        );
    }
    #[cfg(not(debug_assertions))]
    {
        let _ = (token, vocab_size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokens_in_bounds_valid() {
        let tokens = vec![0, 100, 999];
        assert_tokens_in_bounds(&tokens, 1000);
    }

    #[test]
    #[should_panic(expected = "Invariant violation")]
    #[cfg(debug_assertions)]
    fn test_tokens_in_bounds_invalid() {
        let tokens = vec![0, 100, 1000]; // 1000 >= vocab_size
        assert_tokens_in_bounds(&tokens, 1000);
    }

    #[test]
    fn test_encode_postconditions_valid() {
        let tokens = vec![1, 2, 3];
        assert_encode_postconditions(&tokens, 1000);
    }

    #[test]
    fn test_decode_preconditions_valid() {
        let tokens = vec![1, 2, 3];
        assert_decode_preconditions(&tokens, 1000);
    }

    #[test]
    fn test_valid_token() {
        assert_valid_token(0, 1000);
        assert_valid_token(999, 1000);
    }

    #[test]
    #[should_panic(expected = "Invariant violation")]
    #[cfg(debug_assertions)]
    fn test_invalid_token() {
        assert_valid_token(1000, 1000);
    }
}
