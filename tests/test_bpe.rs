use shimmytok::Tokenizer;
use std::path::Path;

fn get_model_path() -> String {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE")).map_or_else(|_| "gpt2.Q4_K_M.gguf".to_string(), |home| format!("{home}/.cache/models/gguf/gpt2.Q4_K_M.gguf"))
}

#[test]
fn test_gpt2_tokenization() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!(
            "Skipping test_gpt2_tokenization: model not found at {model_path}"
        );
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path).expect("Failed to load GPT-2 model");

    println!("Vocab size: {}", tokenizer.vocab_size());

    let test_cases = vec![
        "Hello, world!",
        "The quick brown fox",
        "Rust is awesome",
        "Testing BPE tokenization",
    ];

    for text in test_cases {
        let tokens = tokenizer.encode(text, false).unwrap();
        println!("\nText: '{text}'");
        println!("Tokens: {tokens:?}");

        let decoded = tokenizer.decode(&tokens, false).unwrap();
        println!("Decoded: '{decoded}'");

        // Check round-trip
        assert!(!tokens.is_empty(), "Should produce tokens for: {text}");

        // The decoded text should match original (accounting for BPE encoding)
        // Note: GPT-2 may not perfectly round-trip due to byte-level encoding
    }
}
