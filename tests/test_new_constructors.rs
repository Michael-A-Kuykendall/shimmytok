//! Tests for the new `from_reader`, `from_bytes`, and `chat_template` APIs.

use shimmytok::Tokenizer;
use std::io::Write;
use tempfile::NamedTempFile;

// ── helpers ──────────────────────────────────────────────────────────────────

/// Build a minimal valid GGUF v3 byte payload containing a tiny vocabulary.
/// Used to exercise constructors without needing a real model file.
fn minimal_gguf_bytes() -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();

    // Magic + version (v3)
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes()); // version
    buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    buf.extend_from_slice(&1u64.to_le_bytes()); // metadata_count = 1 kv pair

    // Single kv: "tokenizer.ggml.tokens" = ["<unk>", "hello", "world"]
    let key = "tokenizer.ggml.tokens";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key.as_bytes());

    // value type 9 = array, element type 8 = string
    buf.extend_from_slice(&9u32.to_le_bytes()); // array type
    buf.extend_from_slice(&8u32.to_le_bytes()); // element type = string
    buf.extend_from_slice(&3u64.to_le_bytes()); // 3 elements

    for tok in &["<unk>", "hello", "world"] {
        buf.extend_from_slice(&(tok.len() as u64).to_le_bytes());
        buf.extend_from_slice(tok.as_bytes());
    }

    buf
}

/// Extend the minimal fixture with the BOOL-array metadata that
/// embeddinggemma emits in GGUF v3 files (GitHub issue #1).
fn gguf_with_bool_array_metadata() -> Vec<u8> {
    let mut buf = minimal_gguf_bytes();

    // GGUF header layout: magic (4), version (4), tensor count (8), metadata count (8).
    buf[16..24].copy_from_slice(&2u64.to_le_bytes());

    let key = "gemma3.attention.sliding_window_pattern";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
    buf.extend_from_slice(&9u32.to_le_bytes()); // value type = array
    buf.extend_from_slice(&7u32.to_le_bytes()); // element type = BOOL
    buf.extend_from_slice(&3u64.to_le_bytes());
    buf.extend_from_slice(&[1, 0, 1]);

    buf
}

// ── from_bytes ────────────────────────────────────────────────────────────────

#[test]
fn test_from_bytes_loads_tokenizer() {
    let bytes = minimal_gguf_bytes();
    let tokenizer = Tokenizer::from_bytes(&bytes).expect("from_bytes should succeed");
    assert!(tokenizer.vocab_size() >= 3);
}

#[test]
fn test_from_bytes_accepts_bool_array_metadata() {
    let tokenizer = Tokenizer::from_bytes(&gguf_with_bool_array_metadata())
        .expect("GGUF BOOL arrays should not prevent tokenizer loading");
    assert_eq!(tokenizer.vocab_size(), 3);
}

#[test]
fn test_from_bytes_invalid_magic() {
    let bad = b"NOTGGUF_DATA_HERE".to_vec();
    assert!(Tokenizer::from_bytes(&bad).is_err());
}

#[test]
fn test_from_bytes_empty() {
    assert!(Tokenizer::from_bytes(&[]).is_err());
}

// ── from_reader ───────────────────────────────────────────────────────────────

#[test]
fn test_from_reader_cursor() {
    use std::io::Cursor;
    let bytes = minimal_gguf_bytes();
    let cursor = Cursor::new(bytes);
    let tokenizer = Tokenizer::from_reader(cursor).expect("from_reader should succeed");
    assert!(tokenizer.vocab_size() >= 3);
}

#[test]
fn test_from_reader_file() {
    let bytes = minimal_gguf_bytes();
    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();

    let file = std::fs::File::open(tmp.path()).unwrap();
    let tokenizer =
        Tokenizer::from_reader(std::io::BufReader::new(file)).expect("from_reader(file)");
    assert!(tokenizer.vocab_size() >= 3);
}

// ── from_bytes vs from_gguf_file parity ──────────────────────────────────────

#[test]
fn test_from_bytes_parity_with_from_gguf_file() {
    let bytes = minimal_gguf_bytes();
    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();

    let from_file = Tokenizer::from_gguf_file(tmp.path()).expect("from_gguf_file");
    let from_bytes = Tokenizer::from_bytes(&bytes).expect("from_bytes");

    assert_eq!(from_file.vocab_size(), from_bytes.vocab_size());
    assert_eq!(from_file.model_type(), from_bytes.model_type());
    assert_eq!(from_file.bos_token(), from_bytes.bos_token());
    assert_eq!(from_file.eos_token(), from_bytes.eos_token());
}

// ── chat_template ─────────────────────────────────────────────────────────────

#[test]
fn test_chat_template_absent_when_not_in_gguf() {
    // Our minimal GGUF has no chat_template key
    let bytes = minimal_gguf_bytes();
    let tokenizer = Tokenizer::from_bytes(&bytes).unwrap();
    assert!(tokenizer.chat_template().is_none());
}

/// Build a GGUF that includes a `tokenizer.chat_template` kv pair.
fn gguf_with_chat_template(template: &str) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();

    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    buf.extend_from_slice(&2u64.to_le_bytes()); // 2 kv pairs

    // kv 1: tokenizer.ggml.tokens — must have enough entries to satisfy
    // the debug invariant (BOS defaults to 1, EOS to 2, so need ≥ 3 tokens)
    let key1 = "tokenizer.ggml.tokens";
    buf.extend_from_slice(&(key1.len() as u64).to_le_bytes());
    buf.extend_from_slice(key1.as_bytes());
    buf.extend_from_slice(&9u32.to_le_bytes()); // array
    buf.extend_from_slice(&8u32.to_le_bytes()); // string elements
    buf.extend_from_slice(&3u64.to_le_bytes()); // 3 tokens
    for tok in &["<unk>", "<s>", "</s>"] {
        buf.extend_from_slice(&(tok.len() as u64).to_le_bytes());
        buf.extend_from_slice(tok.as_bytes());
    }

    // kv 2: tokenizer.chat_template
    let key2 = "tokenizer.chat_template";
    buf.extend_from_slice(&(key2.len() as u64).to_le_bytes());
    buf.extend_from_slice(key2.as_bytes());
    buf.extend_from_slice(&8u32.to_le_bytes()); // type = string
    buf.extend_from_slice(&(template.len() as u64).to_le_bytes());
    buf.extend_from_slice(template.as_bytes());

    buf
}
#[test]
fn test_chat_template_present_when_in_gguf() {
    let template = "{% for msg in messages %}{{ msg.role }}: {{ msg.content }}\n{% endfor %}";
    let bytes = gguf_with_chat_template(template);
    let tokenizer = Tokenizer::from_bytes(&bytes).unwrap();
    assert_eq!(tokenizer.chat_template(), Some(template));
}

#[test]
fn test_chat_template_returns_str_not_owned() {
    // Verify it's a &str borrow, not a clone — lifetime check via type inference
    let bytes = gguf_with_chat_template("{{ messages }}");
    let tokenizer = Tokenizer::from_bytes(&bytes).unwrap();
    let tmpl: Option<&str> = tokenizer.chat_template();
    assert!(tmpl.is_some());
}
