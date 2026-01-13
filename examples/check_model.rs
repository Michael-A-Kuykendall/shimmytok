//! Quick model checker - validates shimmytok against llama.cpp tokenize output

use shimmytok::Tokenizer;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: check_model <model.gguf> [prompt]");
        std::process::exit(1);
    }
    
    let model_path = &args[1];
    let prompt = args.get(2).map(|s| s.as_str()).unwrap_or("Hello");
    
    println!("Loading: {}", model_path);
    
    match Tokenizer::from_gguf_file(model_path) {
        Ok(tokenizer) => {
            println!("Model type: {}", tokenizer.model_type());
            println!("Pre-type: {:?}", tokenizer.pre_type());
            println!("Vocab size: {}", tokenizer.vocab_size());
            println!("BOS token: {}", tokenizer.bos_token());
            println!("EOS token: {}", tokenizer.eos_token());
            println!();
            
            // Encode without special tokens (matches llama.cpp --no-bos)
            match tokenizer.encode(prompt, false) {
                Ok(tokens) => {
                    println!("Prompt: {:?}", prompt);
                    println!("Tokens (no BOS): {:?}", tokens);
                    
                    // Also show with BOS for comparison
                    if let Ok(tokens_with_bos) = tokenizer.encode(prompt, true) {
                        println!("Tokens (with BOS): {:?}", tokens_with_bos);
                    }
                    
                    // Decode back
                    if let Ok(decoded) = tokenizer.decode(&tokens, false) {
                        println!("Decoded: {:?}", decoded);
                    }
                }
                Err(e) => println!("Encode error: {}", e),
            }
        }
        Err(e) => {
            println!("Load error: {}", e);
            std::process::exit(1);
        }
    }
}
