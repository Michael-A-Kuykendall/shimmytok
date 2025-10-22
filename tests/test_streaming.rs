use shimmytok::Tokenizer;
use std::path::Path;

fn get_model_path() -> String {
    std::env::var("GGUF_MODEL_PATH").unwrap_or_else(|_| {
        dirs::home_dir()
            .map(|h| h.join(".cache/models/gguf/gpt2.Q4_K_M.gguf"))
            .and_then(|p| p.to_str().map(String::from))
            .unwrap_or_else(|| "model.gguf".to_string())
    })
}

#[test]
fn test_decode_single() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!(
            "Skipping test_decode_single: model not found at {}",
            model_path
        );
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path).unwrap();

    // Test decoding individual tokens
    let text = "Hello world";
    let tokens = tokenizer.encode(text, false).unwrap();

    // Decode each token individually and compare to full decode
    let mut decoded_parts = Vec::new();
    for &token in &tokens {
        let part = tokenizer.decode_single(token, false).unwrap();
        decoded_parts.push(part);
    }
    let single_decoded = decoded_parts.join("");

    let full_decoded = tokenizer.decode(&tokens, false).unwrap();

    assert_eq!(
        single_decoded, full_decoded,
        "Single decode should match full decode"
    );
}

#[test]
fn test_decode_single_with_special_tokens() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!(
            "Skipping test_decode_single_with_special_tokens: model not found at {}",
            model_path
        );
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path).unwrap();

    let bos = tokenizer.bos_token();
    let eos = tokenizer.eos_token();

    // With skip_special_tokens=true, should return empty string
    let bos_decoded = tokenizer.decode_single(bos, true).unwrap();
    assert_eq!(bos_decoded, "", "BOS should be empty when skipping special");

    let eos_decoded = tokenizer.decode_single(eos, true).unwrap();
    assert_eq!(eos_decoded, "", "EOS should be empty when skipping special");

    // With skip_special_tokens=false, should return token representation
    let bos_decoded = tokenizer.decode_single(bos, false).unwrap();
    assert!(!bos_decoded.is_empty(), "BOS should have representation");
}

#[test]
fn test_token_to_piece() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!(
            "Skipping test_token_to_piece: model not found at {}",
            model_path
        );
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path).unwrap();

    let text = "Hello";
    let tokens = tokenizer.encode(text, false).unwrap();

    // Get token pieces
    for &token in &tokens {
        let piece = tokenizer.token_to_piece(token).unwrap();
        assert!(!piece.is_empty(), "Token piece should not be empty");
    }

    // Test invalid token
    let invalid_token = tokenizer.vocab_size() as u32 + 100;
    let result = tokenizer.token_to_piece(invalid_token);
    assert!(result.is_err(), "Should error on invalid token");
}

#[test]
fn test_token_type() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!(
            "Skipping test_token_type: model not found at {}",
            model_path
        );
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path).unwrap();

    // BOS token should be Control or similar
    let bos_type = tokenizer.token_type(tokenizer.bos_token());
    println!("BOS token type: {:?}", bos_type);

    // Regular tokens should be Normal or similar
    let text = "Hello";
    let tokens = tokenizer.encode(text, false).unwrap();
    for &token in &tokens {
        let token_type = tokenizer.token_type(token);
        println!("Token {} type: {:?}", token, token_type);
    }

    // Invalid token should return Undefined
    let invalid_token = tokenizer.vocab_size() as u32 + 100;
    let invalid_type = tokenizer.token_type(invalid_token);
    assert_eq!(
        invalid_type,
        shimmytok::TokenType::Undefined,
        "Invalid token should be Undefined"
    );
}

#[test]
fn test_is_special_token() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!(
            "Skipping test_is_special_token: model not found at {}",
            model_path
        );
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path).unwrap();

    // BOS and EOS should be special
    assert!(
        tokenizer.is_special_token(tokenizer.bos_token()),
        "BOS should be special"
    );
    assert!(
        tokenizer.is_special_token(tokenizer.eos_token()),
        "EOS should be special"
    );

    // Regular content tokens should not be special
    let text = "Hello world";
    let tokens = tokenizer.encode(text, false).unwrap();

    let non_special_count = tokens
        .iter()
        .filter(|&&t| !tokenizer.is_special_token(t))
        .count();
    assert!(
        non_special_count > 0,
        "Should have at least some non-special tokens"
    );

    // Invalid token should return false
    let invalid_token = tokenizer.vocab_size() as u32 + 100;
    assert!(
        !tokenizer.is_special_token(invalid_token),
        "Invalid token should not be special"
    );
}

#[test]
fn test_streaming_simulation() {
    let model_path = get_model_path();
    if !Path::new(&model_path).exists() {
        eprintln!(
            "Skipping test_streaming_simulation: model not found at {}",
            model_path
        );
        return;
    }

    let tokenizer = Tokenizer::from_gguf_file(&model_path).unwrap();

    // Simulate streaming generation
    let text = "Hello world!";
    let tokens = tokenizer.encode(text, true).unwrap(); // With BOS/EOS

    // Stream decode
    let mut streamed = String::new();
    for &token in &tokens {
        let piece = tokenizer.decode_single(token, true).unwrap(); // Skip special
        streamed.push_str(&piece);
    }

    // Full decode
    let full = tokenizer.decode(&tokens, true).unwrap();

    assert_eq!(streamed, full, "Streamed decode should match full decode");
}
