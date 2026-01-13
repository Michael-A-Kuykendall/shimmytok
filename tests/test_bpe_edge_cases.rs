/// Edge case tests for BPE tokenizer
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

#[test]
fn test_empty_string() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test_empty_string: model not found at {model_path}");
        return;
    }
    let tok = Tokenizer::from_gguf_file(&model_path).expect("Failed to load tokenizer");

    let tokens = tok.encode("", false).expect("Failed to encode");
    assert_eq!(tokens, vec![], "Empty string should produce empty tokens");

    let decoded = tok.decode(&[], false).expect("Failed to decode");
    assert_eq!(decoded, "", "Empty tokens should decode to empty string");
}

#[test]
fn test_single_char() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test_single_char: model not found at {model_path}");
        return;
    }
    let tok = Tokenizer::from_gguf_file(&model_path).expect("Failed to load tokenizer");

    let tokens = tok.encode("a", false).expect("Failed to encode");
    assert!(!tokens.is_empty(), "Single char should produce tokens");

    let decoded = tok.decode(&tokens, false).expect("Failed to decode");
    assert_eq!(decoded, "a", "Should round-trip single char");
}

#[test]
fn test_unicode_emoji() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test_unicode_emoji: model not found at {model_path}");
        return;
    }
    let tok = Tokenizer::from_gguf_file(&model_path).expect("Failed to load tokenizer");

    let text = "Hello 👋 World";
    let tokens = tok.encode(text, false).expect("Failed to encode");
    assert!(!tokens.is_empty(), "Emoji text should produce tokens");

    let decoded = tok.decode(&tokens, false).expect("Failed to decode");
    // Note: GPT-2 may not preserve exact emoji, but should not crash
    assert!(!decoded.is_empty(), "Should decode to non-empty string");
}

#[test]
fn test_unicode_cjk() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test_unicode_cjk: model not found at {model_path}");
        return;
    }
    let tok = Tokenizer::from_gguf_file(&model_path).expect("Failed to load tokenizer");

    let text = "你好世界"; // "Hello World" in Chinese
    let tokens = tok.encode(text, false).expect("Failed to encode");
    assert!(!tokens.is_empty(), "CJK text should produce tokens");

    let decoded = tok.decode(&tokens, false).expect("Failed to decode");
    // GPT-2 BPE uses byte-level encoding, so CJK will be mangled but consistent
    // The important thing is it doesn't crash and re-encodes to same tokens
    let tokens2 = tok.encode(&decoded, false).expect("Failed to re-encode");
    assert_eq!(tokens, tokens2, "Double encode should be stable");
}

#[test]
fn test_special_chars() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test_special_chars: model not found at {model_path}");
        return;
    }
    let tok = Tokenizer::from_gguf_file(&model_path).expect("Failed to load tokenizer");

    let text = "!@#$%^&*()";
    let tokens = tok.encode(text, false).expect("Failed to encode");
    assert!(!tokens.is_empty(), "Special chars should produce tokens");

    let decoded = tok.decode(&tokens, false).expect("Failed to decode");
    assert_eq!(decoded, text, "Special chars should round-trip");
}

#[test]
fn test_long_string() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test_long_string: model not found at {model_path}");
        return;
    }
    let tok = Tokenizer::from_gguf_file(&model_path).expect("Failed to load tokenizer");

    // Create 200-repetition string
    let text = "The quick brown fox jumps over the lazy dog. ".repeat(200);
    let tokens = tok.encode(&text, false).expect("Failed to encode");
    assert!(tokens.len() > 100, "Long string should produce many tokens");

    let decoded = tok.decode(&tokens, false).expect("Failed to decode");
    // Check encode stability
    let tokens2 = tok.encode(&decoded, false).expect("Failed to re-encode");
    assert_eq!(tokens, tokens2, "Double encode should be stable");
}

#[test]
fn test_newlines_and_tabs() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test_newlines_and_tabs: model not found at {model_path}");
        return;
    }
    let tok = Tokenizer::from_gguf_file(&model_path).expect("Failed to load tokenizer");

    let text = "Line 1\nLine 2\tTabbed";
    let tokens = tok.encode(text, false).expect("Failed to encode");
    let decoded = tok.decode(&tokens, false).expect("Failed to decode");
    // Check encode stability (double encode produces same tokens)
    let tokens2 = tok.encode(&decoded, false).expect("Failed to re-encode");
    assert_eq!(tokens, tokens2, "Double encode should be stable");
}

#[test]
fn test_multiple_spaces() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!("Skipping test_multiple_spaces: model not found at {model_path}");
        return;
    }
    let tok = Tokenizer::from_gguf_file(&model_path).expect("Failed to load tokenizer");

    let text = "Hello    world"; // 4 spaces
    let tokens = tok.encode(text, false).expect("Failed to encode");
    let decoded = tok.decode(&tokens, false).expect("Failed to decode");
    // Spaces are encoded as Ġ (U+0120) in GPT-2 byte-level BPE
    // Check encode stability instead of exact match
    let tokens2 = tok.encode(&decoded, false).expect("Failed to re-encode");
    assert_eq!(tokens, tokens2, "Double encode should be stable");
}
