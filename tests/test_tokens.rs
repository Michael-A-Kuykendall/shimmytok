use shimmytok::vocab::Vocabulary;

#[test]
fn test_vocabulary_loading() {
    let path = "../aistatepilot-mcp/models/Phi-3-mini-4k-instruct-q4.gguf";

    if std::path::Path::new(path).exists() {
        let vocab = Vocabulary::from_gguf_file(path).unwrap();

        // Check vocabulary size
        println!("Vocabulary has {} tokens", vocab.n_tokens());

        // Look for specific tokens
        for i in 0..100 {
            if let Some(text) = vocab.get_token_text(i) {
                if text.contains("Hello") || text.contains("hello") {
                    println!("Token {i}: '{text}'");
                }
            }
        }

        // Check if " Hello" exists (with space prefix)
        if let Some(id) = vocab.get_token_id(" Hello") {
            println!("' Hello' = token {id}");
        }

        if let Some(id) = vocab.get_token_id("Hello") {
            println!("'Hello' = token {id}");
        }

        // Look for larger token indices
        for i in 15000..15100 {
            if let Some(text) = vocab.get_token_text(i) {
                if text.contains("Hello") || text.contains("hello") {
                    println!("Token {i}: '{text}'");
                }
            }
        }
    }
}
