use shimmytok::{vocab::Vocabulary, Tokenizer};

#[test]
fn test_detailed_merge() {
    let path = "../aistatepilot-mcp/models/Phi-3-mini-4k-instruct-q4.gguf";

    if std::path::Path::new(path).exists() {
        let vocab = Vocabulary::from_gguf_file(path).unwrap();

        // Check what tokens exist for our test
        println!("\nTokens in vocabulary:");

        // Check individual characters
        for ch in ['▁', 'H', 'e', 'l', 'o', 'w', 'r', 'd'] {
            let s = ch.to_string();
            if let Some(id) = vocab.get_token_id(&s) {
                println!(
                    "  '{}' = token {} (score: {})",
                    s,
                    id,
                    vocab.get_token_score(id)
                );
            }
        }

        // Check merged tokens
        let test_tokens = [
            "▁", "▁H", "▁He", "▁Hel", "▁Hell", "▁Hello", "▁w", "▁wo", "▁wor", "▁worl", "▁world",
            "Hello", "world", "He", "llo", "wo", "rld",
        ];

        println!("\nMerged tokens:");
        for token in &test_tokens {
            if let Some(id) = vocab.get_token_id(token) {
                println!(
                    "  '{}' = token {} (score: {})",
                    token,
                    id,
                    vocab.get_token_score(id)
                );
            }
        }

        // Now test encoding
        let tokenizer = Tokenizer::from_gguf_file(path).unwrap();
        let text = "Hello world";
        let tokens = tokenizer.encode(text, false).unwrap();

        println!("\nFinal result:");
        println!("  Input: '{}'", text);
        println!("  Tokens: {:?}", tokens);

        for &id in &tokens {
            if let Some(text) = vocab.get_token_text(id) {
                println!("    {} -> '{}'", id, text);
            }
        }

        // Compare with expected
        println!("\nExpected from llama.cpp: [15043, 3186]");
        println!("  15043 -> '▁Hello'");
        println!("  3186 -> '▁world'");
    }
}
