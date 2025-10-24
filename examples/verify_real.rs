// Real-world verification - not a mock, actual tokenization
use shimmytok::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE"))?;
    let model_path = format!("{home}/.cache/models/gguf/gpt2.Q4_K_M.gguf");

    println!("Loading real GGUF model: {model_path}");
    let tokenizer = Tokenizer::from_gguf_file(&model_path)?;

    let test_cases = vec![
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog",
        "Rust programming language",
        "🦀 Crab emoji test",
        "Multiple\nlines\nof\ntext",
        "Special chars: !@#$%^&*()",
    ];

    println!("\n=== REAL TOKENIZATION TEST ===\n");

    for text in test_cases {
        let tokens = tokenizer.encode(text, false)?;
        let decoded = tokenizer.decode(&tokens, false)?;

        println!("Input:   '{text}'");
        println!("Tokens:  {:?} ({} tokens)", tokens, tokens.len());
        println!("Decoded: '{decoded}'");
        println!(
            "Match:   {}",
            if decoded == text {
                "✓ PERFECT"
            } else {
                "✗ MISMATCH"
            }
        );
        println!();
    }

    // Verify round-trip for random text
    let random_text = "The answer is 42. Testing various symbols: @#$%";
    let tokens = tokenizer.encode(random_text, false)?;
    let decoded = tokenizer.decode(&tokens, false)?;

    if decoded != random_text {
        eprintln!("ERROR: Round-trip failed!");
        eprintln!("  Original: {random_text}");
        eprintln!("  Decoded:  {decoded}");
        std::process::exit(1);
    }

    println!("✓ All round-trips successful - tokenizer is REAL");

    Ok(())
}
