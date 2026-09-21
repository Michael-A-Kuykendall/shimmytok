//! `tokenizer.ggml.add_bos_token` default parity with llama.cpp.
//!
//! llama.cpp: when the key is ABSENT, add_bos_token defaults to true ONLY for
//! SentencePiece tokenizers and false for every other tokenizer family (BPE,
//! WPM, RWKV, UGM); an explicit key always wins. Regression context: the
//! MiniCPM-V 2.6 language GGUF is BPE ("gpt2") without this key, and defaulting
//! it to true poisoned every prompt with a leading BOS.
//!
//! Real-fixture tests target the local model store and skip silently when
//! files are absent (mirrors tests/test_real_models.rs).

use shimmytok::{EncodeOptions, Tokenizer};

// ── in-memory GGUF builder (pattern from tests/test_multimodal_contract.rs) ──

fn push_key(buf: &mut Vec<u8>, key: &str) {
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
}

fn push_string_value(buf: &mut Vec<u8>, value: &str) {
    buf.extend_from_slice(&8u32.to_le_bytes());
    buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
    buf.extend_from_slice(value.as_bytes());
}

fn push_string_array(buf: &mut Vec<u8>, values: &[&str]) {
    buf.extend_from_slice(&9u32.to_le_bytes());
    buf.extend_from_slice(&8u32.to_le_bytes());
    buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for v in values {
        buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
        buf.extend_from_slice(v.as_bytes());
    }
}

fn push_u32_value(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&4u32.to_le_bytes());
    buf.extend_from_slice(&value.to_le_bytes());
}

fn push_bool_value(buf: &mut Vec<u8>, value: bool) {
    buf.extend_from_slice(&7u32.to_le_bytes());
    buf.push(u8::from(value));
}

/// Builds a GGUF v3 payload with a configurable tokenizer model type and an
/// optional `tokenizer.ggml.add_bos_token` key (None = key absent).
fn fixture(model_type: &str, add_bos: Option<bool>) -> Vec<u8> {
    let mut body = Vec::new();
    push_key(&mut body, "tokenizer.ggml.model");
    push_string_value(&mut body, model_type);
    push_key(&mut body, "tokenizer.ggml.tokens");
    push_string_array(
        &mut body,
        if model_type == "gpt2" {
            &["<unk>", "<s>", "</s>", "a", "b", "c", "ab", "abc", "1", "2"][..]
        } else {
            &["<unk>", "<s>", "</s>", "a", "b", "ab"][..]
        },
    );
    if model_type == "gpt2" {
        push_key(&mut body, "tokenizer.ggml.merges");
        push_string_array(&mut body, &["a b", "ab c"]);
    }
    push_key(&mut body, "tokenizer.ggml.unknown_token_id");
    push_u32_value(&mut body, 0);
    push_key(&mut body, "tokenizer.ggml.bos_token_id");
    push_u32_value(&mut body, 1);
    push_key(&mut body, "tokenizer.ggml.eos_token_id");
    push_u32_value(&mut body, 2);
    if let Some(flag) = add_bos {
        push_key(&mut body, "tokenizer.ggml.add_bos_token");
        push_bool_value(&mut body, flag);
    }

    let kv_count = 5usize + usize::from(model_type == "gpt2") + usize::from(add_bos.is_some());

    let mut buf = Vec::with_capacity(24 + body.len());
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    buf.extend_from_slice(&(kv_count as u64).to_le_bytes());
    buf.extend_from_slice(&body);
    buf
}

fn bpe_fixture(add_bos: Option<bool>) -> Vec<u8> {
    fixture("gpt2", add_bos)
}

fn spm_fixture(add_bos: Option<bool>) -> Vec<u8> {
    fixture("llama", add_bos)
}

// ── synthetic matrix ─────────────────────────────────────────────────────────

#[test]
fn bpe_absent_key_defaults_to_no_bos() {
    let tok = Tokenizer::from_bytes(&bpe_fixture(None)).expect("fixture loads");
    let ids = tok.encode("abc", true).expect("encode");
    assert_eq!(
        ids,
        vec![7],
        "BPE with absent add_bos_token must NOT prepend BOS"
    );
}

#[test]
fn bpe_explicit_true_prepends_bos() {
    let tok = Tokenizer::from_bytes(&bpe_fixture(Some(true))).expect("fixture loads");
    let ids = tok.encode("abc", true).expect("encode");
    assert_eq!(ids, vec![1, 7]);
}

#[test]
fn bpe_explicit_false_keeps_no_bos() {
    let tok = Tokenizer::from_bytes(&bpe_fixture(Some(false))).expect("fixture loads");
    let ids = tok.encode("abc", true).expect("encode");
    assert_eq!(ids, vec![7]);
}

#[test]
fn spm_absent_key_defaults_to_bos() {
    let tok = Tokenizer::from_bytes(&spm_fixture(None)).expect("fixture loads");
    let ids = tok.encode("a", true).expect("encode");
    assert_eq!(ids[0], 1, "SPM with absent add_bos_token must prepend BOS");
    assert!(ids.iter().all(|&t| t < 6));
}

#[test]
fn spm_explicit_false_keeps_no_bos() {
    let tok = Tokenizer::from_bytes(&spm_fixture(Some(false))).expect("fixture loads");
    let ids = tok.encode("a", true).expect("encode");
    assert_ne!(ids.first(), Some(&1), "explicit false must not prepend BOS");
}

// ── real fixtures (local model store) ────────────────────────────────────────

const MINICPM_GGUF: &str = "MiniCPM-V-2.6/ggml-model-Q4_K_M.gguf";
const QWEN2_05B_GGUF: &str = "Qwen2/Qwen2-0.5B-Instruct-Q4_K_M/qwen2-0_5b-instruct-q4_k_m.gguf";
const TINYLLAMA_GGUF: &str =
    "Llama/TinyLlama-1.1B-Chat-Q5_K_M/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf";

fn model_path(rel: &str) -> Option<String> {
    let root = std::env::var("SHIMMYTOK_MODEL_ROOT")
        .unwrap_or_else(|_| "/home/michael/models".to_string());
    let p = format!("{root}/{rel}");
    std::path::Path::new(&p).exists().then_some(p)
}

/// The exact rendered prompt captured from the MiniCPM-V GgufJinja path.
const MINICPM_RENDERED_PROMPT: &str = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>What text appears in the center of this screen?<|im_end|>\n<|im_start|>assistant\n";

#[test]
fn minicpm_v26_bpe_absent_key_must_not_prepend_bos() {
    let Some(path) = model_path(MINICPM_GGUF) else {
        eprintln!("skip: MiniCPM GGUF not present");
        return;
    };
    let tok = Tokenizer::from_gguf_file(&path).expect("MiniCPM loads");
    assert_eq!(tok.model_type(), "gpt2");
    assert!(
        !tok.metadata().contains_key("tokenizer.ggml.add_bos_token"),
        "fixture changed: MiniCPM GGUF now carries add_bos_token"
    );

    let plain = tok.encode("Hello", true).expect("encode");
    assert_eq!(
        plain.iter().filter(|&&t| t == tok.bos_token()).count(),
        0,
        "BPE with absent add_bos_token must not prepend BOS"
    );

    let prompt = tok
        .encode_with_options(
            MINICPM_RENDERED_PROMPT,
            &EncodeOptions::with_parse_special(true, true),
        )
        .expect("encode rendered prompt");
    let im_start = tok
        .get_token("<|im_start|>")
        .expect("<|im_start|> in vocab");
    assert_eq!(prompt[0], im_start, "prompt must start with <|im_start|>");
    assert_eq!(
        prompt.iter().filter(|&&t| t == im_start).count(),
        3,
        "exactly three template openers; no duplicated leading BOS"
    );
}

#[test]
fn minicpm_v26_wrapper_tokens_exist_in_vocab() {
    let Some(path) = model_path(MINICPM_GGUF) else {
        eprintln!("skip: MiniCPM GGUF not present");
        return;
    };
    let tok = Tokenizer::from_gguf_file(&path).expect("MiniCPM loads");
    for piece in ["<image>", "</image>", "<slice>", "</slice>"] {
        let id = tok
            .get_token(piece)
            .unwrap_or_else(|| panic!("wrapper piece {piece} missing from vocab"));
        println!("wrapper piece {piece} -> id {id}");
    }
}

#[test]
fn tinyllama_spm_keeps_bos() {
    let Some(path) = model_path(TINYLLAMA_GGUF) else {
        eprintln!("skip: TinyLlama GGUF not present");
        return;
    };
    let tok = Tokenizer::from_gguf_file(&path).expect("TinyLlama loads");
    assert_eq!(tok.model_type(), "llama");
    let ids = tok.encode("Hello", true).expect("encode");
    assert_eq!(ids[0], tok.bos_token(), "SPM default must keep BOS");
}

#[test]
fn qwen2_comparator_with_explicit_key_unchanged() {
    let Some(path) = model_path(QWEN2_05B_GGUF) else {
        eprintln!("skip: Qwen2 comparator GGUF not present");
        return;
    };
    let tok = Tokenizer::from_gguf_file(&path).expect("Qwen2 loads");
    assert!(matches!(tok.model_type(), "gpt2" | "qwen2"));
    assert!(
        tok.metadata().contains_key("tokenizer.ggml.add_bos_token"),
        "fixture changed: comparator GGUF no longer carries add_bos_token"
    );
    let ids = tok.encode("Hello", true).expect("encode");
    assert_ne!(
        ids[0],
        tok.bos_token(),
        "explicit false must not prepend BOS"
    );
}
