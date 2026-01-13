//! Basic usage example for shimmytok
//!
//! Run with: cargo run --example basic -- path/to/model.gguf

use shimmytok::Tokenizer;
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Usage: cargo run --example basic -- <model.gguf>");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  cargo run --example basic -- ~/.cache/models/llama-3.gguf");
        std::process::exit(1);
    }
    
    let model_path = &args[1];
    println!("Loading tokenizer from: {}", model_path);
    
    // Load tokenizer from GGUF file
    let tokenizer = Tokenizer::from_gguf_file(model_path)?;
    
    // Print tokenizer info
    println!();
    println!("Tokenizer loaded!");
    println!("  Model type: {}", tokenizer.model_type());
    println!("  Pre-tokenizer: {:?}", tokenizer.pre_type());
    println!("  Vocab size: {}", tokenizer.vocab_size());
    println!("  BOS token: {}", tokenizer.bos_token());
    println!("  EOS token: {}", tokenizer.eos_token());
    
    // Test encoding
    let test_text = "Hello, world! This is a test of shimmytok. 🦀";
    println!();
    println!("Test text: {:?}", test_text);
    
    let tokens = tokenizer.encode(test_text, false)?;
    println!("Tokens ({}): {:?}", tokens.len(), tokens);
    
    // Test decoding
    let decoded = tokenizer.decode(&tokens, false)?;
    println!("Decoded: {:?}", decoded);
    
    // Verify round-trip
    if decoded == test_text {
        println!("✅ Round-trip successful!");
    } else {
        println!("⚠️  Round-trip mismatch (may be expected for some tokenizers)");
    }
    
    // Demo streaming decode
    println!();
    println!("Streaming decode:");
    print!("  ");
    for token_id in &tokens {
        let piece = tokenizer.decode_single(*token_id, false)?;
        print!("{}", piece);
    }
    println!();
    
    Ok(())
}
