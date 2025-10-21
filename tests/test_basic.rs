use shimmytok::Tokenizer;

#[test]
fn test_load_gguf() {
    // Try to load a GGUF file if one exists
    let paths = [
        "../aistatepilot-mcp/models/Phi-3-mini-4k-instruct-q4.gguf",
        "../libshimmy/models/phi-2.Q4_K_M.gguf",
        "../archive/shimmy-original/shimmy/models/tinyllama.gguf",
    ];

    for path in &paths {
        if std::path::Path::new(path).exists() {
            println!("Testing with: {}", path);

            match Tokenizer::from_gguf_file(path) {
                Ok(tokenizer) => {
                    // Test basic tokenization
                    let text = "Hello world";
                    match tokenizer.encode(text, false) {
                        Ok(tokens) => {
                            println!(
                                "  Encoded '{}' to {} tokens: {:?}",
                                text,
                                tokens.len(),
                                tokens
                            );

                            // Try to decode back
                            match tokenizer.decode(&tokens, false) {
                                Ok(decoded) => {
                                    println!("  Decoded back to: '{}'", decoded);
                                }
                                Err(e) => {
                                    println!("  Decode failed: {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            println!("  Encode failed: {}", e);
                        }
                    }
                }
                Err(e) => {
                    println!("  Failed to load: {}", e);
                }
            }

            break;
        }
    }
}
