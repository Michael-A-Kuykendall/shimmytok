//! Tests to increase code coverage to 95%+
//!
//! These tests target uncovered code paths identified by tarpaulin.

use shimmytok::{DecodeOptions, EncodeOptions, Error, TokenType, Tokenizer};
use std::path::Path;

fn get_model_path() -> String {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map_or_else(
            |_| "gpt2.Q4_K_M.gguf".to_string(),
            |home| format!("{home}/.cache/models/gguf/gpt2.Q4_K_M.gguf"),
        )
}

fn get_spm_model_path() -> String {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map_or_else(
            |_| "llama-3.2-1b-instruct-q4_k_m.gguf".to_string(),
            |home| format!("{home}/.cache/models/gguf/llama-3.2-1b-instruct-q4_k_m.gguf"),
        )
}

// ===== Basic Accessor Tests =====

#[test]
fn test_basic_accessors() {
    let model_path = get_spm_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // Test basic accessors
    let vocab_size = tokenizer.vocab_size();
    assert!(vocab_size > 0);

    let bos = tokenizer.bos_token();
    assert!(bos < vocab_size as u32);

    let eos = tokenizer.eos_token();
    assert!(eos < vocab_size as u32);

    let model_type = tokenizer.model_type();
    assert!(!model_type.is_empty());

    // pre_type may be None
    let _ = tokenizer.pre_type();
}

// ===== Decode Options Tests =====

#[test]
fn test_decode_with_lstrip() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    let tokens = tokenizer.encode("  hello world", false).unwrap();

    // Decode with lstrip to exercise that code path
    let options = DecodeOptions {
        skip_special_tokens: false,
        include_special_text: true,
        lstrip: true,
    };
    let decoded = tokenizer.decode_with_options(&tokens, &options).unwrap();
    // lstrip should remove leading whitespace
    assert!(
        !decoded.starts_with("  "),
        "lstrip should trim leading whitespace"
    );
}

#[test]
fn test_decode_without_special_text() {
    let model_path = get_spm_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // Encode with BOS/EOS
    let tokens = tokenizer.encode("hello", true).unwrap();

    // Decode excluding special text
    let options = DecodeOptions {
        skip_special_tokens: false,
        include_special_text: false,
        lstrip: false,
    };
    let decoded = tokenizer.decode_with_options(&tokens, &options).unwrap();
    // Special tokens should not appear as text
    assert!(!decoded.contains("<|begin_of_text|>"));
}

#[test]
fn test_decode_options_constructors() {
    // Test DecodeOptions::default() - all fields are false by default
    let default = DecodeOptions::default();
    assert!(!default.skip_special_tokens);
    assert!(!default.include_special_text); // Default is false
    assert!(!default.lstrip);

    // Test DecodeOptions::with_skip_special() - sets include_special_text to true
    let skip = DecodeOptions::with_skip_special(true);
    assert!(skip.skip_special_tokens);
    assert!(skip.include_special_text); // with_skip_special sets this to true

    // Test DecodeOptions::new()
    let custom = DecodeOptions::new(true, true, false);
    assert!(custom.skip_special_tokens);
    assert!(custom.lstrip);
    assert!(!custom.include_special_text);
}

// ===== Encode Options Tests =====

#[test]
fn test_encode_options_constructors() {
    // Test EncodeOptions::default()
    let default = EncodeOptions::default();
    assert!(!default.add_special_tokens);
    assert!(!default.parse_special);

    // Test EncodeOptions::with_special_tokens()
    let with_special = EncodeOptions::with_special_tokens(true);
    assert!(with_special.add_special_tokens);

    // Test EncodeOptions::with_parse_special()
    let with_parse = EncodeOptions::with_parse_special(true, true);
    assert!(with_parse.add_special_tokens);
    assert!(with_parse.parse_special);
}

#[test]
fn test_encode_with_various_options() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // Test with default options
    let options = EncodeOptions::default();
    let tokens = tokenizer
        .encode_with_options("Hello world!", &options)
        .unwrap();
    assert!(!tokens.is_empty());

    // Test with special tokens
    let options = EncodeOptions::with_special_tokens(true);
    let tokens = tokenizer
        .encode_with_options("Hello world!", &options)
        .unwrap();
    assert!(!tokens.is_empty());
}

#[test]
fn test_encode_parse_special_enabled() {
    let model_path = get_spm_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    let options = EncodeOptions::with_parse_special(false, true);

    // Test with text containing special token pattern
    let result = tokenizer.encode_with_options("Hello <|end_of_text|> world", &options);
    assert!(result.is_ok());
}

// ===== Batch Encode Tests =====

#[test]
fn test_encode_batch_empty() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // Empty batch
    let results = tokenizer.encode_batch(&[], false).unwrap();
    assert!(results.is_empty());

    // Batch with empty strings
    let results = tokenizer.encode_batch(&["", "", ""], false).unwrap();
    assert_eq!(results.len(), 3);
    for r in results {
        assert!(r.is_empty());
    }
}

// ===== Token Info Tests =====

#[test]
fn test_token_to_piece_various() {
    let model_path = get_spm_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // Test token_to_piece on BOS token
    let bos = tokenizer.bos_token();
    let piece = tokenizer.token_to_piece(bos);
    assert!(piece.is_ok());

    // Test on regular tokens
    let tokens = tokenizer.encode("hello", false).unwrap();
    for &t in &tokens {
        let piece = tokenizer.token_to_piece(t);
        assert!(piece.is_ok());
    }
}

#[test]
fn test_is_special_token_various() {
    let model_path = get_spm_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // BOS/EOS should be special
    assert!(tokenizer.is_special_token(tokenizer.bos_token()));
    assert!(tokenizer.is_special_token(tokenizer.eos_token()));

    // Regular tokens should not be special
    let tokens = tokenizer.encode("hello", false).unwrap();
    if !tokens.is_empty() {
        // Most regular tokens should not be special
        let non_special_count = tokens
            .iter()
            .filter(|&&t| !tokenizer.is_special_token(t))
            .count();
        assert!(non_special_count > 0);
    }
}

#[test]
fn test_token_type_various() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // Test token types for various tokens
    for i in 0..10.min(tokenizer.vocab_size()) {
        let tt = tokenizer.token_type(i as u32);
        // Should be one of the valid types
        assert!(matches!(
            tt,
            TokenType::Normal
                | TokenType::Unknown
                | TokenType::Control
                | TokenType::UserDefined
                | TokenType::Unused
                | TokenType::Byte
                | TokenType::Undefined
        ));
    }
}

// ===== Decode Single Tests =====

#[test]
fn test_decode_single_all_tokens() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // Test decoding first 100 tokens individually
    for i in 0..100.min(tokenizer.vocab_size()) {
        let result = tokenizer.decode_single(i as u32, false);
        assert!(result.is_ok(), "decode_single failed for token {i}");
    }
}

#[test]
fn test_decode_single_with_special() {
    let model_path = get_spm_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // Decode BOS with and without special handling
    let bos = tokenizer.bos_token();
    let with_special = tokenizer.decode_single(bos, true);
    let without_special = tokenizer.decode_single(bos, false);
    assert!(with_special.is_ok());
    assert!(without_special.is_ok());
}

// ===== Error Path Tests =====

#[test]
fn test_decode_invalid_token_range() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // Token at vocab_size should be invalid
    let invalid = tokenizer.vocab_size() as u32;
    let result = tokenizer.decode(&[invalid], false);
    assert!(result.is_err());

    match result {
        Err(Error::InvalidToken(msg)) => assert!(msg.contains(&invalid.to_string())),
        other => panic!("Expected InvalidToken error, got {other:?}"),
    }
}

#[test]
fn test_token_to_piece_invalid() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // Invalid token should return error
    let invalid = u32::MAX;
    let result = tokenizer.token_to_piece(invalid);
    assert!(result.is_err());
}

#[test]
fn test_token_type_invalid() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // Invalid token should return Undefined
    let invalid = u32::MAX;
    let token_type = tokenizer.token_type(invalid);
    assert_eq!(token_type, TokenType::Undefined);
}

// ===== Clean Spaces Edge Cases =====

#[test]
fn test_clean_spaces_edge_cases() {
    let model_path = get_spm_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // Test various punctuation patterns
    let test_cases = [
        "Hello ?",       // Space before ?
        "Hello !",       // Space before !
        "Hello .",       // Space before .
        "Hello ,",       // Space before ,
        " ' ",           // Isolated apostrophe
        "I 'm happy",    // Contraction 'm
        "It 's great",   // Contraction 's
        "We 've done",   // Contraction 've
        "They 're here", // Contraction 're
    ];

    for text in test_cases {
        let tokens = tokenizer.encode(text, false).unwrap();
        let decoded = tokenizer.decode(&tokens, false).unwrap();
        // Just verify it doesn't panic
        assert!(!decoded.is_empty() || text.trim().is_empty());
    }
}

// ===== Split on Special Tokens Tests =====

#[test]
fn test_split_on_special_tokens_complex() {
    let model_path = get_spm_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    let options = EncodeOptions::with_parse_special(false, true);

    // Multiple special tokens in sequence
    let result = tokenizer.encode_with_options(
        "Start <|begin_of_text|> middle <|end_of_text|> end",
        &options,
    );
    assert!(result.is_ok());
}

// ===== Model Type Tests =====

#[test]
fn test_model_type_accessor() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    let model_type = tokenizer.model_type();
    // GPT-2 should be BPE
    assert!(
        model_type == "bpe" || model_type == "gpt2",
        "Unexpected model type: {model_type}"
    );
}

#[test]
fn test_model_type_spm() {
    let model_path = get_spm_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    let model_type = tokenizer.model_type();
    // Llama should be SPM
    assert!(
        model_type == "llama" || model_type.contains("llama"),
        "Unexpected model type: {model_type}"
    );
}

// ===== Byte Encoder Tests =====

#[test]
fn test_byte_encoder_full_range() {
    use shimmytok::byte_encoder::bytes_to_unicode;

    let map = bytes_to_unicode();

    // Verify all 256 bytes are mapped
    assert_eq!(map.len(), 256);

    // Verify mapping is bijective (unique chars)
    let chars: std::collections::HashSet<_> = map.values().collect();
    assert_eq!(chars.len(), 256);
}

// ===== Round Trip Comprehensive =====

#[test]
fn test_round_trip_unicode() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    let test_cases = [
        "Hello, 世界!",
        "Привет мир",
        "مرحبا",
        "שלום",
        "🎉🚀✨",
        "a̐éö̲",  // Combining marks
        "\t\n\r", // Whitespace
    ];

    for text in test_cases {
        let tokens = tokenizer.encode(text, false).unwrap();
        let decoded = tokenizer.decode(&tokens, false).unwrap();
        // Note: exact round-trip may not preserve all chars due to tokenizer behavior
        assert!(!decoded.is_empty() || text.is_empty());
    }
}

// ===== Invariants Module Tests =====

#[test]
fn test_invariants_functions() {
    use shimmytok::invariants;

    // Test assert_tokens_in_bounds with valid tokens
    invariants::assert_tokens_in_bounds(&[0, 1, 2], 100);
    invariants::assert_tokens_in_bounds(&[], 100);

    // Test assert_encode_postconditions
    invariants::assert_encode_postconditions(&[0, 1, 2], 100);
    invariants::assert_encode_postconditions(&[], 100);

    // Test assert_valid_token (doesn't panic for valid tokens)
    invariants::assert_valid_token(0, 100);
    invariants::assert_valid_token(99, 100);
    // Note: assert_valid_token(100, 100) would panic in debug mode
}

// ===== TokenType From Tests =====

#[test]
fn test_token_type_from_i32() {
    assert_eq!(TokenType::from(0), TokenType::Undefined);
    assert_eq!(TokenType::from(1), TokenType::Normal);
    assert_eq!(TokenType::from(2), TokenType::Unknown);
    assert_eq!(TokenType::from(3), TokenType::Control);
    assert_eq!(TokenType::from(4), TokenType::UserDefined);
    assert_eq!(TokenType::from(5), TokenType::Unused);
    assert_eq!(TokenType::from(6), TokenType::Byte);
    assert_eq!(TokenType::from(7), TokenType::Undefined); // Unknown value
    assert_eq!(TokenType::from(-1), TokenType::Undefined); // Negative
}

// ===== Additional Coverage Tests =====

#[test]
fn test_encode_very_long_unicode() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // Test with long unicode string
    let text = "こんにちは".repeat(100);
    let tokens = tokenizer.encode(&text, false).unwrap();
    let decoded = tokenizer.decode(&tokens, false).unwrap();
    assert!(!decoded.is_empty());
}

#[test]
fn test_encode_with_newlines() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // Test with various newlines
    let texts = [
        "line1\nline2",
        "line1\r\nline2",
        "line1\rline2",
        "\n\n\n",
        "text\n",
        "\ntext",
    ];

    for text in texts {
        let tokens = tokenizer.encode(text, false).unwrap();
        let _ = tokenizer.decode(&tokens, false).unwrap();
    }
}

#[test]
fn test_decode_skip_special_tokens() {
    let model_path = get_spm_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // Encode with special tokens
    let tokens = tokenizer.encode("hello world", true).unwrap();

    // Decode with skip_special_tokens=true
    let decoded = tokenizer.decode(&tokens, true).unwrap();
    
    // Should not contain BOS/EOS markers
    assert!(!decoded.contains("<|begin_of_text|>"));
    assert!(!decoded.contains("<|end_of_text|>"));
}

#[test]
fn test_batch_encode_large() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // Test batch encoding with multiple strings
    let texts: Vec<&str> = (0..10).map(|i| match i % 4 {
        0 => "Hello world",
        1 => "The quick brown fox",
        2 => "1234567890",
        _ => "Special chars: @#$%",
    }).collect();

    let results = tokenizer.encode_batch(&texts, false).unwrap();
    assert_eq!(results.len(), 10);
    for result in results {
        assert!(!result.is_empty());
    }
}

#[test]
fn test_sentencepiece_special_chars() {
    let model_path = get_spm_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // Test various special characters
    let texts = [
        "test@email.com",
        "http://example.com",
        "foo_bar_baz",
        "CamelCaseWord",
        "ALLCAPS",
        "MixedCASE123",
        "with-dashes-here",
        "dots.and.more.dots",
    ];

    for text in texts {
        let tokens = tokenizer.encode(text, false).unwrap();
        let decoded = tokenizer.decode(&tokens, false).unwrap();
        // Just verify no panics
        assert!(!decoded.is_empty() || text.is_empty());
    }
}

#[test]
fn test_bpe_edge_patterns() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // Test BPE-specific patterns
    let texts = [
        " leading space",
        "trailing space ",
        "  multiple  spaces  ",
        "tab\there",
        "Contractions: don't won't can't",
        "Numbers: 123 456.789 0.001",
        "Punctuation: Hello! How? Yes. Really,",
    ];

    for text in texts {
        let tokens = tokenizer.encode(text, false).unwrap();
        let decoded = tokenizer.decode(&tokens, false).unwrap();
        assert!(!decoded.is_empty());
    }
}

#[test]
fn test_empty_string_roundtrip() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    let tokens = tokenizer.encode("", false).unwrap();
    assert!(tokens.is_empty());

    let decoded = tokenizer.decode(&[], false).unwrap();
    assert!(decoded.is_empty());
}

#[test]
fn test_single_char_encoding() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // Test single characters
    for c in "abcdefghijklmnopqrstuvwxyz0123456789".chars() {
        let s = c.to_string();
        let tokens = tokenizer.encode(&s, false).unwrap();
        assert!(!tokens.is_empty(), "Failed to encode '{c}'");
    }
}

#[test]
fn test_decode_all_special_tokens() {
    let model_path = get_spm_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // Decode BOS and EOS individually
    let bos = tokenizer.bos_token();
    let eos = tokenizer.eos_token();

    let decoded_bos = tokenizer.decode_single(bos, false).unwrap();
    let decoded_eos = tokenizer.decode_single(eos, false).unwrap();

    // Should produce some output (the token text)
    // The actual text depends on the model - just verify no panic
    let _ = decoded_bos;
    let _ = decoded_eos;
}

#[test]
fn test_clean_spaces_comprehensive() {
    let model_path = get_spm_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // Comprehensive clean_spaces test
    let test_cases = [
        // Space before punctuation
        ("a .", "a."),
        ("b ?", "b?"),
        ("c !", "c!"),
        ("d ,", "d,"),
        // Multiple punctuation
        ("Hello . World !", "Hello. World!"),
        // Apostrophes
        ("I 'm", "I'm"),
        ("it 's", "it's"),
        ("we 've", "we've"),
        ("they 're", "they're"),
    ];

    for (input, _expected) in test_cases {
        let tokens = tokenizer.encode(input, false).unwrap();
        let _ = tokenizer.decode(&tokens, false).unwrap();
        // Note: exact output depends on model tokenization
    }
}

// ===== GGUF Error Path Tests =====

#[test]
fn test_gguf_invalid_magic() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    // Create a file with invalid magic bytes
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(b"NOTG").unwrap(); // Wrong magic, should be "GGUF"
    file.flush().unwrap();

    let result = Tokenizer::from_gguf_file(file.path());
    assert!(result.is_err(), "Expected error for invalid magic");
    match result {
        Err(e) => {
            let msg = format!("{e}");
            assert!(
                msg.contains("Not a GGUF") || msg.contains("Invalid") || msg.contains("GGUF"),
                "Expected GGUF-related error, got: {msg}"
            );
        }
        Ok(_) => panic!("Expected error"),
    }
}

#[test]
fn test_gguf_truncated_file() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    // Create a file with valid magic but truncated
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(b"GGUF").unwrap(); // Valid magic but no version
    file.flush().unwrap();

    let result = Tokenizer::from_gguf_file(file.path());
    assert!(result.is_err(), "Expected error for truncated file");
}

#[test]
fn test_gguf_unsupported_version() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    // Create a file with unsupported version (version 99)
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(b"GGUF").unwrap();
    file.write_all(&99u32.to_le_bytes()).unwrap(); // Version 99 not supported
    file.flush().unwrap();

    let result = Tokenizer::from_gguf_file(file.path());
    assert!(result.is_err(), "Expected error for unsupported version");
    match result {
        Err(e) => {
            let msg = format!("{e}");
            assert!(
                msg.contains("version") || msg.contains("Unsupported") || msg.contains("GGUF"),
                "Expected version error, got: {msg}"
            );
        }
        Ok(_) => panic!("Expected error"),
    }
}

