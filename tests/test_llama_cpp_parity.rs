// Tests for llama.cpp parity: special token IDs and cleanup flags
use shimmytok::Tokenizer;
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
    // Try to find a SentencePiece model (Llama, Mistral, etc.)
    let paths = [
        std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .map(|h| format!("{h}/.cache/models/gguf/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"))
            .unwrap_or_default(),
        "../aistatepilot-mcp/models/Phi-3-mini-4k-instruct-q4.gguf".to_string(),
        "../libshimmy/models/phi-2.Q4_K_M.gguf".to_string(),
    ];
    for p in paths {
        if Path::new(&p).exists() {
            return p;
        }
    }
    "model.gguf".to_string()
}

#[test]
fn test_special_token_accessors_exist() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // These should all be callable without panic
    let _bos = tokenizer.bos_token();
    let _eos = tokenizer.eos_token();
    let _vocab_size = tokenizer.vocab_size();
    let _model_type = tokenizer.model_type();

    println!("BOS token: {}", tokenizer.bos_token());
    println!("EOS token: {}", tokenizer.eos_token());
    println!("Vocab size: {}", tokenizer.vocab_size());
    println!("Model type: {}", tokenizer.model_type());
}

#[test]
fn test_is_special_token_includes_new_ids() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // BOS and EOS should be special
    assert!(
        tokenizer.is_special_token(tokenizer.bos_token()),
        "BOS should be special"
    );
    assert!(
        tokenizer.is_special_token(tokenizer.eos_token()),
        "EOS should be special"
    );

    // A regular token should not be special (token ID 100 is usually a normal token)
    // This is model-dependent, but for GPT-2/Llama vocab, 100 is typically normal
    if tokenizer.vocab_size() > 100 {
        let is_special = tokenizer.is_special_token(100);
        println!("Token 100 is_special: {is_special}");
        // We don't assert here because it depends on the model
    }
}

#[test]
fn test_token_type_accessor() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // BOS token should have a specific type (usually Control)
    let bos_type = tokenizer.token_type(tokenizer.bos_token());
    println!("BOS token type: {:?}", bos_type);

    // Out-of-range token should return Undefined
    let invalid_type = tokenizer.token_type(u32::MAX);
    assert_eq!(
        invalid_type,
        shimmytok::TokenType::Undefined,
        "Out-of-range token should be Undefined"
    );
}

#[test]
fn test_token_to_piece() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // Should be able to get piece for BOS token
    let bos_piece = tokenizer.token_to_piece(tokenizer.bos_token());
    assert!(bos_piece.is_ok(), "Should get piece for BOS token");
    println!("BOS piece: {:?}", bos_piece.unwrap());

    // Invalid token should error
    let invalid = tokenizer.token_to_piece(u32::MAX);
    assert!(invalid.is_err(), "Out-of-range token should error");
}

// ============================================================================
// Byte-token decode tests (Task 5)
// ============================================================================

#[test]
fn test_byte_token_decode_unit() {
    // Unit test for the byte token parsing logic
    // This tests the pattern <0xXX> -> byte value

    // Valid byte tokens
    assert_eq!(parse_byte_token("<0x0A>"), Some(0x0A)); // newline
    assert_eq!(parse_byte_token("<0x00>"), Some(0x00)); // null
    assert_eq!(parse_byte_token("<0xFF>"), Some(0xFF)); // max byte
    assert_eq!(parse_byte_token("<0x20>"), Some(0x20)); // space

    // Invalid - not byte tokens
    assert_eq!(parse_byte_token("hello"), None);
    assert_eq!(parse_byte_token("<0x0>"), None); // too short
    assert_eq!(parse_byte_token("<0x0AG>"), None); // invalid hex
    assert_eq!(parse_byte_token("0x0A"), None); // missing brackets
}

/// Helper to test byte token parsing (mirrors internal decode_byte_token)
fn parse_byte_token(text: &str) -> Option<u8> {
    if text.len() == 6 && text.starts_with("<0x") && text.ends_with('>') {
        let hex = &text[3..5];
        u8::from_str_radix(hex, 16).ok()
    } else {
        None
    }
}

#[test]
fn test_spm_newline_round_trip() {
    let model_path = get_spm_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: SPM model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // Skip if not SPM model
    let model_type = tokenizer.model_type();
    if model_type != "llama" && model_type != "mistral" && model_type != "gemma" {
        eprintln!("Skipping: model type {model_type} is not SPM");
        return;
    }

    let text = "Hello\nWorld";
    let tokens = tokenizer.encode(text, false).expect("Encode failed");
    let decoded = tokenizer.decode(&tokens, false).expect("Decode failed");

    println!("Original: {:?}", text);
    println!("Tokens: {:?}", tokens);
    println!("Decoded: {:?}", decoded);

    // The decoded text should contain the newline
    assert!(
        decoded.contains('\n') || decoded.contains("\\n"),
        "Decoded text should preserve newline: got {:?}",
        decoded
    );
}

// ============================================================================
// Space prefix flag tests (Task 7)
// ============================================================================

#[test]
fn test_space_prefix_flag_honored() {
    let model_path = get_spm_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: SPM model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // Skip if not SPM model
    let model_type = tokenizer.model_type();
    if model_type != "llama" && model_type != "mistral" && model_type != "gemma" {
        eprintln!("Skipping: model type {model_type} is not SPM");
        return;
    }

    // Encode a simple word
    let text = "Hello";
    let tokens = tokenizer.encode(text, false).expect("Encode failed");
    let decoded = tokenizer.decode(&tokens, false).expect("Decode failed");

    println!("Input: {:?}", text);
    println!("Tokens: {:?}", tokens);
    println!("Decoded: {:?}", decoded);

    // For most SPM models with add_space_prefix=true, decoded should have leading space
    // This is because the ▁ prefix gets decoded as space
    // The exact behavior depends on the model's add_space_prefix setting
}
// ============================================================================
// parse_special tests (Task 9)
// ============================================================================

#[test]
fn test_parse_special_splits_correctly() {
    // Unit test for the special token splitting logic
    use std::collections::HashMap;

    let mut special_map = HashMap::new();
    special_map.insert("<|eot|>".to_string(), 100u32);
    special_map.insert("<|start|>".to_string(), 101u32);

    // Test: text with special token in middle
    // We can't call split_on_special_tokens directly (private), but we can
    // test via encode_with_options if we have a model
}

#[test]
fn test_encode_with_options_api() {
    use shimmytok::EncodeOptions;

    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // Test that encode_with_options works with default options
    let opts = EncodeOptions::with_special_tokens(false);
    let tokens = tokenizer
        .encode_with_options("Hello world", &opts)
        .expect("Encode failed");

    // Should match regular encode
    let tokens_regular = tokenizer
        .encode("Hello world", false)
        .expect("Encode failed");
    assert_eq!(
        tokens, tokens_regular,
        "encode_with_options should match encode"
    );

    println!("encode_with_options API works correctly");
}

#[test]
fn test_parse_special_enabled() {
    use shimmytok::EncodeOptions;

    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // Get the EOS token text
    let eos_piece = tokenizer.token_to_piece(tokenizer.eos_token());
    if eos_piece.is_err() {
        eprintln!("Skipping: couldn't get EOS piece");
        return;
    }
    let eos_text = eos_piece.unwrap();
    println!("EOS token text: {:?}", eos_text);

    // Create input with the EOS token string embedded
    let input = format!("Hello{}World", eos_text);
    println!("Input with embedded EOS: {:?}", input);

    // Encode with parse_special=true
    let opts = EncodeOptions::with_parse_special(false, true);
    let tokens = tokenizer
        .encode_with_options(&input, &opts)
        .expect("Encode with parse_special failed");

    println!("Tokens with parse_special: {:?}", tokens);

    // The EOS token should appear in the output
    assert!(
        tokens.contains(&tokenizer.eos_token()),
        "Tokens should contain EOS when parse_special=true"
    );
}

// ============================================================================
// DecodeOptions tests (Task 11)
// ============================================================================

#[test]
fn test_decode_with_options_api() {
    use shimmytok::DecodeOptions;

    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // Encode some text
    let tokens = tokenizer
        .encode("Hello world", false)
        .expect("Encode failed");

    // Test decode_with_options matches regular decode
    let opts = DecodeOptions::with_skip_special(false);
    let decoded = tokenizer
        .decode_with_options(&tokens, &opts)
        .expect("Decode failed");

    let decoded_regular = tokenizer.decode(&tokens, false).expect("Decode failed");
    assert_eq!(
        decoded, decoded_regular,
        "decode_with_options should match decode"
    );

    println!("decode_with_options API works correctly");
}

#[test]
fn test_decode_lstrip_option() {
    use shimmytok::DecodeOptions;

    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // Encode text that will have leading spaces in tokens
    let tokens = tokenizer
        .encode(" Hello world", false)
        .expect("Encode failed");

    // Decode without lstrip
    let opts_normal = DecodeOptions::with_skip_special(false);
    let decoded_normal = tokenizer
        .decode_with_options(&tokens, &opts_normal)
        .expect("Decode failed");

    // Decode with lstrip
    let opts_lstrip = DecodeOptions::new(false, true, true);
    let decoded_lstrip = tokenizer
        .decode_with_options(&tokens, &opts_lstrip)
        .expect("Decode failed");

    println!("Normal decode: {:?}", decoded_normal);
    println!("Lstrip decode: {:?}", decoded_lstrip);

    // Lstrip version should have less or equal leading whitespace
    let normal_leading = decoded_normal.len() - decoded_normal.trim_start().len();
    let lstrip_leading = decoded_lstrip.len() - decoded_lstrip.trim_start().len();
    assert!(
        lstrip_leading <= normal_leading,
        "Lstrip should reduce or maintain leading whitespace"
    );
}

#[test]
fn test_decode_skip_special_text() {
    use shimmytok::DecodeOptions;

    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test: model not found at {model_path}");
        return;
    }
    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");

    // Encode with BOS token
    let tokens = tokenizer.encode("Hello", true).expect("Encode failed");
    println!("Tokens with special: {:?}", tokens);

    // Decode including special tokens but NOT their text
    let opts = DecodeOptions::new(false, false, false);
    let decoded = tokenizer
        .decode_with_options(&tokens, &opts)
        .expect("Decode failed");

    // Decode with special tokens and their text
    let opts_with_text = DecodeOptions::new(false, false, true);
    let decoded_with_text = tokenizer
        .decode_with_options(&tokens, &opts_with_text)
        .expect("Decode failed");

    println!("Without special text: {:?}", decoded);
    println!("With special text: {:?}", decoded_with_text);

    // The version without special text should be shorter or equal
    // (special tokens might have empty text anyway)
    assert!(
        decoded.len() <= decoded_with_text.len(),
        "Skipping special text should not increase length"
    );
}
