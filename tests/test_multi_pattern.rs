//! Unit tests for multi-pattern sequential tokenization logic

use shimmytok::Tokenizer;

#[test]
fn test_two_pattern_splitting() {
    // Test 2-pattern model (StarCoder)
    let model_path = std::env::var("HOME").unwrap() + "/.cache/models/gguf/starcoder2-3b-q4_k_m.gguf";
    
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping: model not found at {model_path}");
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");
    
    // Text with numbers and letters - should split by number pattern first
    let text = "abc123def456";
    let tokens = tokenizer.encode(text, false).expect("Encoding failed");
    
    assert!(!tokens.is_empty(), "Should produce tokens");
    
    // Verify round-trip
    let decoded = tokenizer.decode(&tokens, false).expect("Decoding failed");
    assert_eq!(decoded, text, "Round-trip should preserve text");
}

#[test]
fn test_six_pattern_splitting() {
    // Test 6-pattern model (DeepSeek-LLM)
    let model_path = std::env::var("HOME").unwrap() + "/.cache/models/gguf/deepseek-llm-7b-chat-q4_k_m.gguf";
    
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping: model not found at {model_path}");
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");
    
    // Complex text with newlines, punctuation, CJK, and numbers
    let text = "Hello!\n你好 123";
    let tokens = tokenizer.encode(text, false).expect("Encoding failed");
    
    assert!(!tokens.is_empty(), "Should produce tokens");
    
    // Verify round-trip
    let decoded = tokenizer.decode(&tokens, false).expect("Decoding failed");
    assert_eq!(decoded, text, "Round-trip should preserve text");
}

#[test]
fn test_gap_preservation() {
    // Test that non-matching regions are preserved as separate fragments
    let model_path = std::env::var("HOME").unwrap() + "/.cache/models/gguf/gpt2-q4_k_m.gguf";
    
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping: model not found at {model_path}");
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");
    
    // Text with clear word boundaries
    let text = "word1 word2 word3";
    let tokens = tokenizer.encode(text, false).expect("Encoding failed");
    
    // Should preserve spaces between words
    let decoded = tokenizer.decode(&tokens, false).expect("Decoding failed");
    assert_eq!(decoded, text, "Spaces should be preserved");
}

#[test]
fn test_default_fallback_pattern() {
    // Test models without pre-tokenizer metadata use DEFAULT (3-pattern)
    // This is the critical fix from v0.6.0
    
    let model_path = std::env::var("HOME").unwrap() + "/.cache/models/gguf/phi-2-q4_k_m.gguf";
    
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping: model not found at {model_path}");
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");
    
    // Test text that exercises all 3 DEFAULT patterns
    let text = "Hello123!";
    let tokens = tokenizer.encode(text, false).expect("Encoding failed");
    
    assert!(!tokens.is_empty(), "Should produce tokens");
    
    // Verify round-trip
    let decoded = tokenizer.decode(&tokens, false).expect("Decoding failed");
    assert_eq!(decoded, text, "Round-trip should preserve text");
}

#[test]
fn test_pattern_with_no_matches() {
    // Pattern that doesn't match anything should pass text through
    let model_path = std::env::var("HOME").unwrap() + "/.cache/models/gguf/gpt2-q4_k_m.gguf";
    
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping: model not found at {model_path}");
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");
    
    // Text with only ASCII letters (no special chars, no numbers)
    let text = "abcdefghijklmnop";
    let tokens = tokenizer.encode(text, false).expect("Encoding failed");
    
    assert!(!tokens.is_empty(), "Should produce tokens even with simple text");
    
    let decoded = tokenizer.decode(&tokens, false).expect("Decoding failed");
    assert_eq!(decoded, text, "Simple text should round-trip correctly");
}

#[test]
fn test_empty_fragments() {
    // Test handling of empty strings after pattern splitting
    let model_path = std::env::var("HOME").unwrap() + "/.cache/models/gguf/gpt2-q4_k_m.gguf";
    
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping: model not found at {model_path}");
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");
    
    // Multiple consecutive spaces
    let text = "word1    word2";
    let tokens = tokenizer.encode(text, false).expect("Encoding failed");
    
    assert!(!tokens.is_empty(), "Should handle multiple spaces");
    
    let decoded = tokenizer.decode(&tokens, false).expect("Decoding failed");
    assert_eq!(decoded, text, "Multiple spaces should be preserved");
}

#[test]
fn test_unicode_in_multi_pattern() {
    // Test Unicode handling across multiple patterns
    let model_path = std::env::var("HOME").unwrap() + "/.cache/models/gguf/qwen2-7b-instruct-q4_k_m.gguf";
    
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping: model not found at {model_path}");
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");
    
    // Mix of ASCII, CJK, emoji
    let text = "Hello 你好 🌍";
    let tokens = tokenizer.encode(text, false).expect("Encoding failed");
    
    assert!(!tokens.is_empty(), "Should handle Unicode");
    
    let decoded = tokenizer.decode(&tokens, false).expect("Decoding failed");
    assert_eq!(decoded, text, "Unicode should round-trip correctly");
}
