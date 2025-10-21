use shimmytok::Tokenizer;
use std::process::Command;

#[test]
fn test_against_llama_cpp() {
    let model_path = "../aistatepilot-mcp/models/Phi-3-mini-4k-instruct-q4.gguf";
    let llama_tokenize = "../llama.cpp/build/bin/Release/llama-tokenize.exe";

    if !std::path::Path::new(model_path).exists() || !std::path::Path::new(llama_tokenize).exists()
    {
        println!("Skipping test - model or llama-tokenize not found");
        return;
    }

    let test_cases = vec![
        "Hello world",
        "The quick brown fox",
        "1234",
        "Hello, world!",
        "This is a test.",
        "🦀 Rust",
        "Multiple  spaces",
        "New\nlines\nhere",
    ];

    let tokenizer = Tokenizer::from_gguf_file(model_path).unwrap();

    for test_text in &test_cases {
        // Get llama.cpp output
        let output = Command::new(llama_tokenize)
            .args(&["-m", model_path, "--prompt", test_text, "--no-bos"])
            .output()
            .expect("Failed to run llama-tokenize");

        let llama_output = String::from_utf8_lossy(&output.stdout);

        // Parse token IDs from llama.cpp output
        let mut llama_tokens = Vec::new();
        for line in llama_output.lines() {
            if let Some(arrow_pos) = line.find(" -> ") {
                if let Ok(token) = line[..arrow_pos].trim().parse::<u32>() {
                    llama_tokens.push(token);
                }
            }
        }

        // Get our output
        let our_tokens = tokenizer.encode(test_text, false).unwrap();

        // Compare
        if our_tokens != llama_tokens {
            println!("MISMATCH for '{}':", test_text);
            println!("  llama.cpp: {:?}", llama_tokens);
            println!("  shimmytok: {:?}", our_tokens);
        } else {
            println!("✓ MATCH for '{}': {:?}", test_text, our_tokens);
        }
    }
}
