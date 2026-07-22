//! Portable API-compatibility parity baseline.
//!
//! These tests run on a clean checkout with NO model files: every tokenizer is
//! built from an in-memory GGUF fixture (see `tests/common`). They lock in the
//! current, correct behaviour of the public API so later refactors (prepared
//! BPE state, optional Rayon, deterministic batch dispatch) can be proven
//! byte-for-byte identical.
//!
//! Coverage maps to the guarantees in `docs/API_STABILITY.md`:
//! - `Tokenizer` is `Send + Sync` (compile-time + runtime)
//! - Single-pattern and multi-pattern BPE encode outputs are stable
//! - `encode_batch` output order matches input and equals individual `encode`
//! - Empty and single-character inputs
//! - encode/decode round-trip

mod common;

use common::{bpe_gpt2_fixture, bpe_starcoder_fixture};
use shimmytok::Tokenizer;

// ── Send + Sync ─────────────────────────────────────────────────────────────

/// Compile-time proof that `Tokenizer` is `Send + Sync`. If this ever regresses
/// the crate will fail to compile here rather than silently breaking consumers
/// (e.g. airframe) that share a tokenizer across threads.
#[test]
fn tokenizer_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Tokenizer>();
}

/// Runtime proof: a shared tokenizer can encode concurrently from many threads.
#[test]
fn tokenizer_shared_across_threads() {
    use std::sync::Arc;
    use std::thread;

    let tok = Arc::new(Tokenizer::from_bytes(&bpe_gpt2_fixture()).unwrap());
    let expected = tok.encode("abc", false).unwrap();

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let tok = Arc::clone(&tok);
            let expected = expected.clone();
            thread::spawn(move || {
                for _ in 0..100 {
                    assert_eq!(tok.encode("abc", false).unwrap(), expected);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
}

// ── Single-pattern BPE (GPT-2 style) ────────────────────────────────────────

#[test]
fn gpt2_encode_vectors() {
    let tok = Tokenizer::from_bytes(&bpe_gpt2_fixture()).unwrap();

    // Merge chain: a+b => ab, ab+c => abc.
    assert_eq!(tok.encode("abc", false).unwrap(), vec![7]);
    // No applicable merge: individual byte tokens in order.
    assert_eq!(tok.encode("acb", false).unwrap(), vec![3, 5, 4]);
    // Single merge.
    assert_eq!(tok.encode("ab", false).unwrap(), vec![6]);
    // Single character.
    assert_eq!(tok.encode("a", false).unwrap(), vec![3]);
    // Empty input.
    assert_eq!(tok.encode("", false).unwrap(), Vec::<u32>::new());
}

#[test]
fn gpt2_round_trip() {
    let tok = Tokenizer::from_bytes(&bpe_gpt2_fixture()).unwrap();
    for text in ["abc", "acb", "ab", "a"] {
        let ids = tok.encode(text, false).unwrap();
        let decoded = tok.decode(&ids, false).unwrap();
        assert_eq!(decoded, text, "round-trip mismatch for {text:?}");
    }
}

// ── Multi-pattern BPE (StarCoder style) ─────────────────────────────────────

#[test]
fn starcoder_multipattern_vectors() {
    let tok = Tokenizer::from_bytes(&bpe_starcoder_fixture()).unwrap();

    // First pattern (\p{N}) splits digits individually.
    assert_eq!(tok.encode("12", false).unwrap(), vec![3, 4]);
    // Letters stay together and merge.
    assert_eq!(tok.encode("ab", false).unwrap(), vec![7]);
    // Mixed: digit split from letter.
    assert_eq!(tok.encode("1a", false).unwrap(), vec![3, 5]);
}

// ── encode_batch order + equivalence ────────────────────────────────────────

#[test]
fn batch_matches_individual_and_preserves_order() {
    let tok = Tokenizer::from_bytes(&bpe_gpt2_fixture()).unwrap();
    let texts = ["abc", "", "acb", "ab", "a"];

    let batch = tok.encode_batch(&texts, false).unwrap();

    assert_eq!(batch.len(), texts.len());
    for (i, text) in texts.iter().enumerate() {
        let individual = tok.encode(text, false).unwrap();
        assert_eq!(batch[i], individual, "batch[{i}] mismatch for {text:?}");
    }

    // Explicit order check against known vectors.
    assert_eq!(batch[0], vec![7]);
    assert_eq!(batch[1], Vec::<u32>::new());
    assert_eq!(batch[2], vec![3, 5, 4]);
}

#[test]
fn batch_empty_input() {
    let tok = Tokenizer::from_bytes(&bpe_gpt2_fixture()).unwrap();
    let batch = tok.encode_batch(&[], false).unwrap();
    assert!(batch.is_empty());
}

// ── deterministic batch dispatch + errors ───────────────────────────────────

/// A batch large enough (in items AND bytes) to cross the parallel dispatch
/// threshold must produce exactly the same results as individual encodes, in
/// order. Each input is padded so the batch comfortably exceeds the internal
/// byte gate, forcing the parallel backend under the default feature set.
#[test]
fn large_batch_matches_individual() {
    let tok = Tokenizer::from_bytes(&bpe_gpt2_fixture()).unwrap();
    // 64 inputs of ~64 bytes each => ~4 KB total, above the byte threshold.
    let unit = "abcacbab".repeat(8);
    let variants = [unit.clone(), format!("{unit}a"), format!("{unit}c"), unit];
    let texts: Vec<&str> = (0..64).map(|i| variants[i % 4].as_str()).collect();

    let batch = tok.encode_batch(&texts, false).unwrap();
    assert_eq!(batch.len(), texts.len());
    for (i, text) in texts.iter().enumerate() {
        assert_eq!(batch[i], tok.encode(text, false).unwrap(), "index {i}");
    }
}

/// When several inputs fail, `encode_batch` must return the error from the
/// LOWEST failing index — deterministically, regardless of backend. We exploit
/// the fact that oversized-input errors embed the offending byte length, so we
/// can identify *which* input's error surfaced.
///
/// This is a large-batch case (above the threshold) so it exercises the
/// parallel backend under the default feature set.
#[test]
fn batch_returns_lowest_index_error() {
    use shimmytok::MAX_INPUT_SIZE;

    let tok = Tokenizer::from_bytes(&bpe_gpt2_fixture()).unwrap();

    // Two distinct oversized inputs with different lengths.
    let first_bad_len = MAX_INPUT_SIZE + 1;
    let second_bad_len = MAX_INPUT_SIZE + 500;
    let first_bad = "a".repeat(first_bad_len);
    let second_bad = "b".repeat(second_bad_len);

    // Pad with valid inputs so the batch crosses the parallel threshold, with
    // the first failure at index 10 and a second failure at index 20.
    let mut texts: Vec<&str> = vec!["a"; 32];
    texts[10] = &first_bad;
    texts[20] = &second_bad;

    let err = tok.encode_batch(&texts, false).unwrap_err();
    let msg = err.to_string();

    // Must be the lowest-index failure (index 10 → first_bad_len), never index 20.
    assert!(
        msg.contains(&first_bad_len.to_string()),
        "expected lowest-index error ({first_bad_len} bytes), got: {msg}"
    );
    assert!(
        !msg.contains(&second_bad_len.to_string()),
        "must not surface the higher-index error, got: {msg}"
    );
}

/// The same lowest-index guarantee must hold for a small (sequential) batch.
#[test]
fn small_batch_returns_lowest_index_error() {
    use shimmytok::MAX_INPUT_SIZE;

    let tok = Tokenizer::from_bytes(&bpe_gpt2_fixture()).unwrap();
    let first_bad_len = MAX_INPUT_SIZE + 1;
    let second_bad_len = MAX_INPUT_SIZE + 500;
    let first_bad = "a".repeat(first_bad_len);
    let second_bad = "b".repeat(second_bad_len);

    // Small batch (below threshold → sequential backend).
    let texts: Vec<&str> = vec!["a", &first_bad, &second_bad];

    let err = tok.encode_batch(&texts, false).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains(&first_bad_len.to_string()),
        "expected lowest-index error ({first_bad_len} bytes), got: {msg}"
    );
    assert!(!msg.contains(&second_bad_len.to_string()), "got: {msg}");
}

// ── get_token exact lookup ──────────────────────────────────────────────────

#[test]
fn get_token_exact_lookup() {
    let tok = Tokenizer::from_bytes(&bpe_gpt2_fixture()).unwrap();

    // Known pieces return their vocabulary IDs.
    assert_eq!(tok.get_token("<unk>"), Some(0));
    assert_eq!(tok.get_token("a"), Some(3));
    assert_eq!(tok.get_token("abc"), Some(7));

    // Missing pieces return None.
    assert_eq!(tok.get_token("zzz"), None);
    assert_eq!(tok.get_token(""), None);
}

#[test]
fn get_token_is_exact_no_normalization() {
    let tok = Tokenizer::from_bytes(&bpe_gpt2_fixture()).unwrap();

    // "a" exists; case/whitespace variants must NOT match (no normalization).
    assert_eq!(tok.get_token("a"), Some(3));
    assert_eq!(tok.get_token("A"), None);
    assert_eq!(tok.get_token(" a"), None);
    assert_eq!(tok.get_token("a "), None);
}

#[test]
fn get_token_round_trips_with_token_to_piece() {
    let tok = Tokenizer::from_bytes(&bpe_gpt2_fixture()).unwrap();
    for id in 0u32..8 {
        let piece = tok.token_to_piece(id).unwrap();
        assert_eq!(
            tok.get_token(&piece),
            Some(id),
            "round-trip failed for {id}"
        );
    }
}
