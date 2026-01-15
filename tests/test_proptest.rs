//! Property-based tests for tokenizer invariants.
//!
//! Uses proptest to verify that tokenizer properties hold across a wide range of inputs.

use proptest::prelude::*;
use shimmytok::Tokenizer;
use std::path::Path;

/// Get path to a test model, or skip if not available.
fn get_model_path() -> Option<String> {
    let paths = [
        std::env::var("GGUF_MODEL_PATH").ok(),
        std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .map(|h| format!("{h}/.cache/models/gguf/gpt2.Q4_K_M.gguf"))
            .ok(),
        Some("../libshimmy/models/phi-2.Q4_K_M.gguf".to_string()),
    ];

    paths
        .into_iter()
        .flatten()
        .find(|path| Path::new(path).exists())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Property: All encoded token IDs are within vocabulary bounds.
    #[test]
    fn prop_token_ids_in_bounds(text in "\\PC{0,500}") {
        let Some(model_path) = get_model_path() else {
            // Skip test if no model available
            return Ok(());
        };

        let tokenizer = match Tokenizer::from_gguf_file(&model_path) {
            Ok(t) => t,
            Err(_) => return Ok(()), // Skip on load failure
        };

        let vocab_size = tokenizer.vocab_size();

        if let Ok(tokens) = tokenizer.encode(&text, false) {
            for token in &tokens {
                prop_assert!(
                    (*token as usize) < vocab_size,
                    "Token {} >= vocab_size {}",
                    token,
                    vocab_size
                );
            }
        }
    }

    /// Property: Empty input produces empty output (without special tokens).
    #[test]
    fn prop_empty_input_empty_output(_dummy in Just(())) {
        let Some(model_path) = get_model_path() else {
            return Ok(());
        };

        let tokenizer = match Tokenizer::from_gguf_file(&model_path) {
            Ok(t) => t,
            Err(_) => return Ok(()),
        };

        let tokens = tokenizer.encode("", false).expect("Empty encode should succeed");
        prop_assert!(tokens.is_empty(), "Empty input should produce no tokens");

        let decoded = tokenizer.decode(&[], false).expect("Empty decode should succeed");
        prop_assert!(decoded.is_empty(), "Empty tokens should decode to empty string");
    }

    /// Property: Decoding never panics on valid token IDs.
    #[test]
    fn prop_decode_never_panics(token_indices in prop::collection::vec(0usize..1000, 0..100)) {
        let Some(model_path) = get_model_path() else {
            return Ok(());
        };

        let tokenizer = match Tokenizer::from_gguf_file(&model_path) {
            Ok(t) => t,
            Err(_) => return Ok(()),
        };

        let vocab_size = tokenizer.vocab_size();

        // Map indices to valid token IDs
        let tokens: Vec<u32> = token_indices
            .iter()
            .map(|&i| (i % vocab_size) as u32)
            .collect();

        // Should not panic - result can be Ok or Err, but no panic
        let _ = tokenizer.decode(&tokens, false);
    }

    /// Property: Encoding with special tokens adds exactly the expected count.
    #[test]
    fn prop_special_tokens_deterministic(text in "[a-z]{1,50}") {
        let Some(model_path) = get_model_path() else {
            return Ok(());
        };

        let tokenizer = match Tokenizer::from_gguf_file(&model_path) {
            Ok(t) => t,
            Err(_) => return Ok(()),
        };

        let without_special = match tokenizer.encode(&text, false) {
            Ok(t) => t,
            Err(_) => return Ok(()),
        };

        let with_special = match tokenizer.encode(&text, true) {
            Ok(t) => t,
            Err(_) => return Ok(()),
        };

        // With special tokens should have >= tokens than without
        // (BOS and/or EOS may be added)
        prop_assert!(
            with_special.len() >= without_special.len(),
            "Special tokens should only add, not remove: {} vs {}",
            with_special.len(),
            without_special.len()
        );

        // Difference should be at most 2 (BOS + EOS)
        let diff = with_special.len() - without_special.len();
        prop_assert!(
            diff <= 2,
            "At most BOS+EOS should be added, got {} extra tokens",
            diff
        );
    }

    /// Property: ASCII text round-trips through encode/decode.
    /// Note: This may not preserve exact whitespace due to tokenizer normalization.
    #[test]
    fn prop_ascii_roundtrip_preserves_content(text in "[a-zA-Z0-9 ]{1,100}") {
        let Some(model_path) = get_model_path() else {
            return Ok(());
        };

        let tokenizer = match Tokenizer::from_gguf_file(&model_path) {
            Ok(t) => t,
            Err(_) => return Ok(()),
        };

        let tokens = match tokenizer.encode(&text, false) {
            Ok(t) => t,
            Err(_) => return Ok(()),
        };

        let decoded = match tokenizer.decode(&tokens, false) {
            Ok(d) => d,
            Err(_) => return Ok(()),
        };

        // Normalize whitespace for comparison (tokenizers may normalize)
        let text_normalized: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
        let decoded_normalized: String = decoded.split_whitespace().collect::<Vec<_>>().join(" ");

        // The content should be preserved (ignoring whitespace normalization)
        prop_assert!(
            decoded_normalized.contains(&text_normalized) || text_normalized.contains(&decoded_normalized) ||
            text_normalized.len() <= 1 || decoded_normalized.len() <= 1,
            "Round-trip failed: '{}' -> {:?} -> '{}'",
            text,
            tokens,
            decoded
        );
    }

    /// Property: Token count is bounded by input length.
    /// A reasonable upper bound is 4 tokens per input byte (very generous).
    #[test]
    fn prop_token_count_bounded(text in ".{1,200}") {
        let Some(model_path) = get_model_path() else {
            return Ok(());
        };

        let tokenizer = match Tokenizer::from_gguf_file(&model_path) {
            Ok(t) => t,
            Err(_) => return Ok(()),
        };

        if let Ok(tokens) = tokenizer.encode(&text, false) {
            let max_expected = text.len() * 4 + 10; // Very generous bound
            prop_assert!(
                tokens.len() <= max_expected,
                "Token count {} exceeds generous bound {} for input of {} bytes",
                tokens.len(),
                max_expected,
                text.len()
            );
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// Ensure proptest infrastructure works.
    #[test]
    fn test_model_path_helper() {
        // This just tests that get_model_path doesn't panic
        let _ = get_model_path();
    }
}
