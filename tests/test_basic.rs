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

#[test]
fn test_encode_batch() {
    let model_path = std::env::var("GGUF_MODEL_PATH").unwrap_or_else(|_| {
        dirs::home_dir()
            .map(|h| h.join(".cache/models/gguf/gpt2.Q4_K_M.gguf"))
            .and_then(|p| p.to_str().map(String::from))
            .unwrap_or_else(|| "model.gguf".to_string())
    });

    if !std::path::Path::new(&model_path).exists() {
        eprintln!(
            "Skipping test_encode_batch: model not found at {}",
            model_path
        );
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path).unwrap();

    let texts = vec!["Hello world", "Goodbye world", "Rust is great"];
    let batch = tokenizer.encode_batch(&texts, false).unwrap();

    // Verify matches sequential
    for (i, text) in texts.iter().enumerate() {
        let sequential = tokenizer.encode(text, false).unwrap();
        assert_eq!(batch[i], sequential, "Batch mismatch for: {}", text);
    }
}
