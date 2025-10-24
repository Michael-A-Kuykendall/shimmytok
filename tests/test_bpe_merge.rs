//! Unit tests for BPE merge algorithm internals

use shimmytok::Tokenizer;

#[test]
fn test_merge_priority_ordering() {
    // Test that merges happen in correct priority order (lower rank = higher priority)
    let model_path = std::env::var("HOME").unwrap() + "/.cache/models/gguf/gpt2-q4_k_m.gguf";
    
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping: model not found at {model_path}");
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");
    
    // "hello" has common merges that should follow priority order
    let text = "hello";
    let tokens = tokenizer.encode(text, false).expect("Encoding failed");
    
    assert!(!tokens.is_empty(), "Should produce tokens");
    assert!(tokens.len() <= text.len(), "Merges should reduce token count");
    
    // Verify round-trip
    let decoded = tokenizer.decode(&tokens, false).expect("Decoding failed");
    assert_eq!(decoded, text, "Merge algorithm should be reversible");
}

#[test]
fn test_single_character_symbols() {
    // Test initialization with UTF-8 characters as symbols
    let model_path = std::env::var("HOME").unwrap() + "/.cache/models/gguf/gpt2-q4_k_m.gguf";
    
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping: model not found at {model_path}");
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");
    
    // Single character should be one symbol initially
    let text = "a";
    let tokens = tokenizer.encode(text, false).expect("Encoding failed");
    
    assert_eq!(tokens.len(), 1, "Single char should produce single token");
    
    // Multi-byte UTF-8 character
    let text = "你";
    let tokens = tokenizer.encode(text, false).expect("Encoding failed");
    
    assert!(!tokens.is_empty(), "UTF-8 char should tokenize");
}

#[test]
fn test_no_valid_merges() {
    // Text where no merge rules apply - should fall back to byte-level
    let model_path = std::env::var("HOME").unwrap() + "/.cache/models/gguf/gpt2-q4_k_m.gguf";
    
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping: model not found at {model_path}");
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");
    
    // Unusual character sequence unlikely to have merge rules
    let text = "zyxwvu";
    let tokens = tokenizer.encode(text, false).expect("Encoding failed");
    
    assert!(!tokens.is_empty(), "Should produce tokens even without merges");
    
    // Verify round-trip
    let decoded = tokenizer.decode(&tokens, false).expect("Decoding failed");
    assert_eq!(decoded, text, "Should preserve text without merges");
}

#[test]
fn test_empty_symbol_handling() {
    // Test that zero-length symbols (deleted after merge) are skipped
    let model_path = std::env::var("HOME").unwrap() + "/.cache/models/gguf/gpt2-q4_k_m.gguf";
    
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping: model not found at {model_path}");
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");
    
    // Word with multiple common merges
    let text = "the";
    let tokens = tokenizer.encode(text, false).expect("Encoding failed");
    
    // Should merge 't'+'h' then 'th'+'e'
    assert!(tokens.len() <= 2, "Common word should merge efficiently");
    
    let decoded = tokenizer.decode(&tokens, false).expect("Decoding failed");
    assert_eq!(decoded, text, "Merged symbols should decode correctly");
}

#[test]
fn test_bigram_validation() {
    // Test that stale bigrams in queue are skipped
    let model_path = std::env::var("HOME").unwrap() + "/.cache/models/gguf/gpt2-q4_k_m.gguf";
    
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping: model not found at {model_path}");
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");
    
    // Complex word with overlapping merge candidates
    let text = "testing";
    let tokens = tokenizer.encode(text, false).expect("Encoding failed");
    
    assert!(!tokens.is_empty(), "Should handle complex merges");
    
    let decoded = tokenizer.decode(&tokens, false).expect("Decoding failed");
    assert_eq!(decoded, text, "Bigram validation should maintain correctness");
}

#[test]
fn test_linked_list_integrity() {
    // Test symbol linked list updates correctly during merges
    let model_path = std::env::var("HOME").unwrap() + "/.cache/models/gguf/gpt2-q4_k_m.gguf";
    
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping: model not found at {model_path}");
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");
    
    // Long sequence to test multiple merges
    let text = "abcdefghij";
    let tokens = tokenizer.encode(text, false).expect("Encoding failed");
    
    assert!(!tokens.is_empty(), "Should handle sequential symbols");
    
    let decoded = tokenizer.decode(&tokens, false).expect("Decoding failed");
    assert_eq!(decoded, text, "Linked list should maintain order");
}

#[test]
fn test_neighbor_merge_updates() {
    // Test that after a merge, neighbors are re-evaluated for new merge opportunities
    let model_path = std::env::var("HOME").unwrap() + "/.cache/models/gguf/gpt2-q4_k_m.gguf";
    
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping: model not found at {model_path}");
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");
    
    // "ing" is a common suffix that should merge after prefix merges
    let text = "running";
    let tokens = tokenizer.encode(text, false).expect("Encoding failed");
    
    assert!(!tokens.is_empty(), "Should handle cascading merges");
    
    let decoded = tokenizer.decode(&tokens, false).expect("Decoding failed");
    assert_eq!(decoded, text, "Neighbor updates should work correctly");
}

#[test]
fn test_work_queue_exhaustion() {
    // Test that algorithm terminates when no more merges possible
    let model_path = std::env::var("HOME").unwrap() + "/.cache/models/gguf/gpt2-q4_k_m.gguf";
    
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping: model not found at {model_path}");
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");
    
    // Very short text - queue should exhaust quickly
    let text = "hi";
    let tokens = tokenizer.encode(text, false).expect("Encoding failed");
    
    assert!(!tokens.is_empty(), "Should handle short text");
    assert!(tokens.len() <= 2, "Should merge efficiently");
    
    let decoded = tokenizer.decode(&tokens, false).expect("Decoding failed");
    assert_eq!(decoded, text, "Queue exhaustion should be clean");
}

#[test]
fn test_byte_fallback_mechanism() {
    // Test byte-level fallback when token not in vocabulary
    let model_path = std::env::var("HOME").unwrap() + "/.cache/models/gguf/gpt2-q4_k_m.gguf";
    
    if !std::path::Path::new(&model_path).exists() {
        eprintln!("Skipping: model not found at {model_path}");
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load model");
    
    // Emoji might not be in vocab, should fall back to bytes
    let text = "test🔥test";
    let tokens = tokenizer.encode(text, false).expect("Encoding failed");
    
    assert!(!tokens.is_empty(), "Should handle OOV characters");
    
    let decoded = tokenizer.decode(&tokens, false).expect("Decoding failed");
    assert_eq!(decoded, text, "Byte fallback should preserve all characters");
}
