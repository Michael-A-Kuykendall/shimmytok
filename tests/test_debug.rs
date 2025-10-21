use shimmytok::{vocab::Vocabulary, Tokenizer};

#[test]
fn test_debug_tokens() {
    let path = "../aistatepilot-mcp/models/Phi-3-mini-4k-instruct-q4.gguf";

    if std::path::Path::new(path).exists() {
        let vocab = Vocabulary::from_gguf_file(path).unwrap();
        let tokenizer = Tokenizer::from_gguf_file(path).unwrap();

        // Check what tokens we're getting
        let text = "Hello world";
        let tokens = tokenizer.encode(text, false).unwrap();

        println!("Input: '{}'", text);
        println!("Tokens: {:?}", tokens);

        for &id in &tokens {
            if let Some(text) = vocab.get_token_text(id) {
                println!("  {} -> '{}'", id, text);
            }
        }

        // Check specific tokens llama.cpp got
        println!("\nLlama.cpp tokens:");
        for id in [15043, 3186] {
            if let Some(text) = vocab.get_token_text(id) {
                println!("  {} -> '{}'", id, text);
            }
        }

        // Check if "▁world" exists
        if let Some(id) = vocab.get_token_id("▁world") {
            println!("\n'▁world' = token {}", id);
        }

        // Search for world-related tokens
        println!("\nSearching for world tokens:");
        for i in 3180..3195 {
            if let Some(text) = vocab.get_token_text(i) {
                if text.contains("world") || text.contains("World") {
                    println!("  {} -> '{}'", i, text);
                }
            }
        }
    }
}
