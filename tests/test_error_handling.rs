// Negative tests - verify error handling actually works
use shimmytok::{Error, Tokenizer};

fn get_model_path() -> String {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(|home| format!("{}/.cache/models/gguf/gpt2.Q4_K_M.gguf", home))
        .unwrap_or_else(|_| "gpt2.Q4_K_M.gguf".to_string())
}

#[test]
fn test_invalid_token_id() {
    let tokenizer = Tokenizer::from_gguf_file(get_model_path()).expect("Failed to load model");

    // GPT-2 has ~50K tokens, so 999999 is invalid
    let invalid_tokens = vec![999999];
    let result = tokenizer.decode(&invalid_tokens, false);

    assert!(result.is_err(), "Should fail on invalid token ID");
    match result {
        Err(Error::InvalidToken(_)) => { /* correct error type */ }
        Err(e) => panic!("Wrong error type: {:?}", e),
        Ok(_) => panic!("Should have failed"),
    }
}

#[test]
fn test_missing_file() {
    let result = Tokenizer::from_gguf_file("/nonexistent/path/fake.gguf");
    assert!(result.is_err(), "Should fail on missing file");
}

#[test]
fn test_very_large_input() {
    let tokenizer = Tokenizer::from_gguf_file(get_model_path()).expect("Failed to load model");

    // Create 11MB input (exceeds MAX_INPUT_SIZE of 10MB)
    let large_text = "a".repeat(11 * 1024 * 1024);
    let result = tokenizer.encode(&large_text, false);

    assert!(result.is_err(), "Should fail on oversized input");
    match result {
        Err(Error::TokenizationFailed(msg)) => {
            assert!(
                msg.contains("too large"),
                "Error should mention size: {}",
                msg
            );
        }
        Err(e) => panic!("Wrong error type: {:?}", e),
        Ok(_) => panic!("Should have failed on 11MB input"),
    }
}

#[test]
fn test_empty_input() {
    let tokenizer = Tokenizer::from_gguf_file(get_model_path()).expect("Failed to load model");

    let tokens = tokenizer
        .encode("", false)
        .expect("Empty string should succeed");

    assert_eq!(tokens.len(), 0, "Empty input should produce no tokens");

    let decoded = tokenizer
        .decode(&[], false)
        .expect("Empty token list should succeed");

    assert_eq!(decoded, "", "Empty tokens should decode to empty string");
}

#[test]
fn test_round_trip_fuzz() {
    let tokenizer = Tokenizer::from_gguf_file(get_model_path()).expect("Failed to load model");

    let test_strings = vec![
        "a",
        "ab",
        "abc",
        " ",
        "  ",
        "   ",
        "\n",
        "\n\n",
        "\r\n",
        "🦀",
        "🦀🦀",
        "emoji🦀test",
        "!@#$%^&*()",
        "CamelCase",
        "snake_case",
        "kebab-case",
        "123",
        "3.14159",
        "-42",
        "Mixed Case 123 Test!",
    ];

    for text in test_strings {
        let tokens = tokenizer
            .encode(text, false)
            .expect(&format!("Failed to encode: {:?}", text));

        let decoded = tokenizer
            .decode(&tokens, false)
            .expect(&format!("Failed to decode: {:?}", tokens));

        assert_eq!(
            text, decoded,
            "Round-trip failed for: {:?}\n  Tokens: {:?}\n  Decoded: {:?}",
            text, tokens, decoded
        );
    }
}

#[test]
fn test_decode_with_special_tokens() {
    let tokenizer = Tokenizer::from_gguf_file(get_model_path()).expect("Failed to load model");

    let text = "Hello world";

    // With special tokens
    let tokens_with_special = tokenizer
        .encode(text, true)
        .expect("Encode with special tokens failed");

    // Without special tokens
    let tokens_without = tokenizer
        .encode(text, false)
        .expect("Encode without special tokens failed");

    // With BOS token enabled, should have more tokens
    // (depends on model config, but test the API works)
    println!("With special: {:?}", tokens_with_special);
    println!("Without special: {:?}", tokens_without);

    // Both should decode successfully
    let decoded_with = tokenizer
        .decode(&tokens_with_special, false)
        .expect("Decode with special failed");
    let decoded_without = tokenizer
        .decode(&tokens_without, false)
        .expect("Decode without special failed");

    println!("Decoded with: {:?}", decoded_with);
    println!("Decoded without: {:?}", decoded_without);
}

#[test]
fn test_max_token_validation() {
    let tokenizer = Tokenizer::from_gguf_file(get_model_path()).expect("Failed to load model");

    // Create input that would produce many tokens
    // Worst case: every character becomes a token
    // "a b c d e..." with spaces = 2 tokens per pair
    let many_chars: String = (0..1000).map(|i| format!("w{} ", i)).collect();

    let result = tokenizer.encode(&many_chars, false);
    assert!(
        result.is_ok(),
        "Should handle 1000-word input: {:?}",
        result.err()
    );

    if let Ok(tokens) = result {
        println!("1000-word input produced {} tokens", tokens.len());
        assert!(
            tokens.len() < shimmytok::MAX_OUTPUT_TOKENS,
            "Should not exceed MAX_OUTPUT_TOKENS"
        );
    }
}
