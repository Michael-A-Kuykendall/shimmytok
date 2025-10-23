use shimmytok::Tokenizer;
use std::process::Command;

/// Test strings for BPE validation
const TEST_STRINGS: &[&str] = &[
    "Hello world",
    "The quick brown fox jumps over the lazy dog",
    "This is a test of the BPE tokenizer",
    "function main() { return 0; }",
    "🦀 Rust programming language",
    "Multiple\nlines\nof\ntext",
    "Numbers: 123 456 789",
    "Special chars: !@#$%^&*()",
    "Mixed CASE and lower case",
    "UTF-8: café, naïve, 日本語",
];

fn run_llama_cpp_tokenize(model_path: &str, text: &str) -> Vec<u32> {
    let output = Command::new("../llama.cpp/build/bin/Release/llama-tokenize.exe")
        .arg("-m")
        .arg(model_path)
        .arg("-p")
        .arg(text)
        .arg("--ids")
        .arg("--no-bos")  // Don't add BOS - we'll test that separately
        .output()
        .expect("Failed to run llama-tokenize");

    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Parse output like "[1, 2, 3]"
    stdout
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .filter_map(|s| s.trim().parse::<u32>().ok())
        .collect()
}

#[test]
#[ignore] // Run with: cargo test test_gpt2_validation -- --ignored
fn test_gpt2_validation() {
    let model_path = std::env::var("HOME").unwrap() + "/.cache/models/gguf/gpt2.Q4_K_M.gguf";
    
    if !std::path::Path::new(&model_path).exists() {
        println!("Skipping: model not found at {}", model_path);
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path)
        .expect("Failed to load tokenizer");

    let mut passed = 0;
    let mut failed = 0;

    for (i, text) in TEST_STRINGS.iter().enumerate() {
        println!("\nTest {}: \"{}\"", i + 1, text);
        
        // Shimmytok
        let shimmy_tokens = tokenizer.encode(text, false)
            .expect("Failed to encode with shimmytok");
        
        // llama.cpp
        let llama_tokens = run_llama_cpp_tokenize(&model_path, text);
        
        if shimmy_tokens == llama_tokens {
            println!("  ✅ MATCH: {:?}", shimmy_tokens);
            passed += 1;
        } else {
            println!("  ❌ MISMATCH:");
            println!("     shimmytok: {:?}", shimmy_tokens);
            println!("     llama.cpp: {:?}", llama_tokens);
            failed += 1;
        }
    }
    
    // Adjust totals for partial run
    let total_tests = TEST_STRINGS.len();

    println!("\n========================================");
    println!("GPT-2 Validation Results:");
    println!("  Passed: {}/{}", passed, total_tests);
    println!("  Failed: {}/{}", failed, total_tests);
    println!("========================================");

    assert_eq!(failed, 0, "Some tests failed - see output above");
}

#[test]
#[ignore]
fn test_qwen2_validation() {
    let model_path = std::env::var("HOME").unwrap() + "/.cache/models/gguf/qwen2-7b-instruct-q4_k_m.gguf";
    
    if !std::path::Path::new(&model_path).exists() {
        println!("Skipping: model not found at {}", model_path);
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path)
        .expect("Failed to load tokenizer");

    let mut passed = 0;
    let mut failed = 0;

    for (i, text) in TEST_STRINGS.iter().enumerate() {
        println!("\nTest {}: \"{}\"", i + 1, text);
        
        let shimmy_tokens = tokenizer.encode(text, false)
            .expect("Failed to encode with shimmytok");
        
        let llama_tokens = run_llama_cpp_tokenize(&model_path, text);
        
        if shimmy_tokens == llama_tokens {
            println!("  ✅ MATCH: {:?}", shimmy_tokens);
            passed += 1;
        } else {
            println!("  ❌ MISMATCH:");
            println!("     shimmytok: {:?}", shimmy_tokens);
            println!("     llama.cpp: {:?}", llama_tokens);
            failed += 1;
        }
    }

    println!("\n========================================");
    println!("Qwen2 Validation Results:");
    println!("  Passed: {}/{}", passed, TEST_STRINGS.len());
    println!("  Failed: {}/{}", failed, TEST_STRINGS.len());
    println!("========================================");

    assert_eq!(failed, 0, "Some tests failed - see output above");
}

#[test]
#[ignore]
fn test_starcoder_validation() {
    let model_path = std::env::var("HOME").unwrap() + "/.cache/models/gguf/starcoder2-3b-Q4_K_M.gguf";
    
    if !std::path::Path::new(&model_path).exists() {
        println!("Skipping: model not found at {}", model_path);
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path)
        .expect("Failed to load tokenizer");

    let mut passed = 0;
    let mut failed = 0;

    for (i, text) in TEST_STRINGS.iter().enumerate() {
        println!("\nTest {}: \"{}\"", i + 1, text);
        
        let shimmy_tokens = tokenizer.encode(text, false)
            .expect("Failed to encode with shimmytok");
        
        let llama_tokens = run_llama_cpp_tokenize(&model_path, text);
        
        if shimmy_tokens == llama_tokens {
            println!("  ✅ MATCH: {:?}", shimmy_tokens);
            passed += 1;
        } else {
            println!("  ❌ MISMATCH:");
            println!("     shimmytok: {:?}", shimmy_tokens);
            println!("     llama.cpp: {:?}", llama_tokens);
            failed += 1;
        }
    }

    println!("\n========================================");
    println!("StarCoder Validation Results:");
    println!("  Passed: {}/{}", passed, TEST_STRINGS.len());
    println!("  Failed: {}/{}", failed, TEST_STRINGS.len());
    println!("========================================");

    assert_eq!(failed, 0, "Some tests failed - see output above");
}

#[test]
#[ignore]
fn test_deepseek_coder_validation() {
    let model_path = std::env::var("HOME").unwrap() + "/.cache/models/gguf/deepseek-coder-6.7b-instruct.Q4_K_M.gguf";
    
    if !std::path::Path::new(&model_path).exists() {
        println!("Skipping: model not found at {}", model_path);
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path)
        .expect("Failed to load tokenizer");

    let mut passed = 0;
    let mut failed = 0;

    for (i, text) in TEST_STRINGS.iter().enumerate() {
        println!("\nTest {}: \"{}\"", i + 1, text);
        
        let shimmy_tokens = tokenizer.encode(text, false)
            .expect("Failed to encode with shimmytok");
        
        let llama_tokens = run_llama_cpp_tokenize(&model_path, text);
        
        if shimmy_tokens == llama_tokens {
            println!("  ✅ MATCH: {:?}", shimmy_tokens);
            passed += 1;
        } else {
            println!("  ❌ MISMATCH:");
            println!("     shimmytok: {:?}", shimmy_tokens);
            println!("     llama.cpp: {:?}", llama_tokens);
            failed += 1;
        }
    }

    println!("\n========================================");
    println!("DeepSeek-Coder Validation Results:");
    println!("  Passed: {}/{}", passed, TEST_STRINGS.len());
    println!("  Failed: {}/{}", failed, TEST_STRINGS.len());
    println!("========================================");

    assert_eq!(failed, 0, "Some tests failed - see output above");
}

#[test]
#[ignore]
fn test_deepseek_llm_validation() {
    let model_path = std::env::var("HOME").unwrap() + "/.cache/models/gguf/deepseek-llm-7b-chat.Q4_K_M.gguf";
    
    if !std::path::Path::new(&model_path).exists() {
        println!("Skipping: model not found at {}", model_path);
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path)
        .expect("Failed to load tokenizer");

    let mut passed = 0;
    let mut failed = 0;

    for (i, text) in TEST_STRINGS.iter().enumerate() {
        println!("\nTest {}: \"{}\"", i + 1, text);
        
        let shimmy_tokens = tokenizer.encode(text, false)
            .expect("Failed to encode with shimmytok");
        
        let llama_tokens = run_llama_cpp_tokenize(&model_path, text);
        
        if shimmy_tokens == llama_tokens {
            println!("  ✅ MATCH: {:?}", shimmy_tokens);
            passed += 1;
        } else {
            println!("  ❌ MISMATCH:");
            println!("     shimmytok: {:?}", shimmy_tokens);
            println!("     llama.cpp: {:?}", llama_tokens);
            failed += 1;
        }
    }

    println!("\n========================================");
    println!("DeepSeek-LLM Validation Results:");
    println!("  Passed: {}/{}", passed, TEST_STRINGS.len());
    println!("  Failed: {}/{}", failed, TEST_STRINGS.len());
    println!("========================================");

    assert_eq!(failed, 0, "Some tests failed - see output above");
}

#[test]
#[ignore]
fn test_phi2_validation() {
    let model_path = std::env::var("HOME").unwrap() + "/.cache/models/gguf/phi-2.Q4_K_M.gguf";
    
    if !std::path::Path::new(&model_path).exists() {
        println!("Skipping: model not found at {}", model_path);
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path)
        .expect("Failed to load tokenizer");

    let mut passed = 0;
    let mut failed = 0;

    for (i, text) in TEST_STRINGS.iter().enumerate() {
        println!("\nTest {}: \"{}\"", i + 1, text);
        
        let shimmy_tokens = tokenizer.encode(text, false)
            .expect("Failed to encode with shimmytok");
        
        let llama_tokens = run_llama_cpp_tokenize(&model_path, text);
        
        if shimmy_tokens == llama_tokens {
            println!("  ✅ MATCH: {:?}", shimmy_tokens);
            passed += 1;
        } else {
            println!("  ❌ MISMATCH:");
            println!("     shimmytok: {:?}", shimmy_tokens);
            println!("     llama.cpp: {:?}", llama_tokens);
            failed += 1;
        }
    }

    println!("\n========================================");
    println!("Phi-2 Validation Results:");
    println!("  Passed: {}/{}", passed, TEST_STRINGS.len());
    println!("  Failed: {}/{}", failed, TEST_STRINGS.len());
    println!("========================================");

    assert_eq!(failed, 0, "Some tests failed - see output above");
}
