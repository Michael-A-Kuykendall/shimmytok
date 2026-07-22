//! Shared test helpers for building in-memory GGUF fixtures.
//!
//! These builders let the test suite construct fully-working tokenizers without
//! any model files on disk, so parity and behaviour tests run on a clean
//! checkout (and in CI) with no external dependencies.
//!
//! The GGUF wire format is documented at
//! <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>. Only the value
//! types the tokenizer loader reads are emitted here.

#![allow(dead_code)]

/// GGUF metadata value-type IDs (subset consumed by the loader).
mod ty {
    pub const U32: u32 = 4;
    pub const BOOL: u32 = 7;
    pub const STRING: u32 = 8;
    pub const ARRAY: u32 = 9;
    pub const I32: u32 = 5;
}

/// Incrementally builds a valid GGUF v3 byte payload with tokenizer metadata.
///
/// Each `with_*` call appends one metadata key-value pair; [`build`](Self::build)
/// finalizes the header with the correct pair count.
#[derive(Default)]
pub struct GgufBuilder {
    kv_count: u64,
    body: Vec<u8>,
}

impl GgufBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn push_key(&mut self, key: &str) {
        self.body
            .extend_from_slice(&(key.len() as u64).to_le_bytes());
        self.body.extend_from_slice(key.as_bytes());
    }

    fn push_str_value(&mut self, s: &str) {
        self.body.extend_from_slice(&(s.len() as u64).to_le_bytes());
        self.body.extend_from_slice(s.as_bytes());
    }

    /// Append a `String` metadata value.
    #[must_use]
    pub fn with_string(mut self, key: &str, value: &str) -> Self {
        self.push_key(key);
        self.body.extend_from_slice(&ty::STRING.to_le_bytes());
        self.push_str_value(value);
        self.kv_count += 1;
        self
    }

    /// Append a `u32` metadata value.
    #[must_use]
    pub fn with_u32(mut self, key: &str, value: u32) -> Self {
        self.push_key(key);
        self.body.extend_from_slice(&ty::U32.to_le_bytes());
        self.body.extend_from_slice(&value.to_le_bytes());
        self.kv_count += 1;
        self
    }

    /// Append a `bool` metadata value.
    #[must_use]
    pub fn with_bool(mut self, key: &str, value: bool) -> Self {
        self.push_key(key);
        self.body.extend_from_slice(&ty::BOOL.to_le_bytes());
        self.body.push(u8::from(value));
        self.kv_count += 1;
        self
    }

    /// Append a string-array metadata value.
    #[must_use]
    pub fn with_string_array(mut self, key: &str, values: &[&str]) -> Self {
        self.push_key(key);
        self.body.extend_from_slice(&ty::ARRAY.to_le_bytes());
        self.body.extend_from_slice(&ty::STRING.to_le_bytes());
        self.body
            .extend_from_slice(&(values.len() as u64).to_le_bytes());
        for v in values {
            self.push_str_value(v);
        }
        self.kv_count += 1;
        self
    }

    /// Append an i32-array metadata value.
    #[must_use]
    pub fn with_i32_array(mut self, key: &str, values: &[i32]) -> Self {
        self.push_key(key);
        self.body.extend_from_slice(&ty::ARRAY.to_le_bytes());
        self.body.extend_from_slice(&ty::I32.to_le_bytes());
        self.body
            .extend_from_slice(&(values.len() as u64).to_le_bytes());
        for v in values {
            self.body.extend_from_slice(&v.to_le_bytes());
        }
        self.kv_count += 1;
        self
    }

    /// Finalize the GGUF v3 payload: magic + version + counts + body.
    #[must_use]
    pub fn build(self) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::with_capacity(24 + self.body.len());
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        buf.extend_from_slice(&self.kv_count.to_le_bytes());
        buf.extend_from_slice(&self.body);
        buf
    }
}

/// A minimal, deterministic GPT-2 style BPE fixture (single pre-tokenization
/// pattern). All token pieces use printable ASCII, which the GPT-2 byte encoder
/// maps to itself, so encode outputs are stable and hand-verifiable.
///
/// Vocabulary (id: piece):
/// - 0: `<unk>`  1: `<s>`  2: `</s>`
/// - 3: `a`  4: `b`  5: `c`  6: `ab`  7: `abc`  8: `1`  9: `2`
///
/// Merges (rank order): `a b` -> `ab`, then `ab c` -> `abc`.
///
/// Expected encodes (add_special = false):
/// - `"abc"` -> `[7]`   (a+b => ab, ab+c => abc)
/// - `"acb"` -> `[3,5,4]` (no applicable merge)
/// - `"ab"`  -> `[6]`
/// - `"a"`   -> `[3]`
/// - `""`    -> `[]`
#[must_use]
pub fn bpe_gpt2_fixture() -> Vec<u8> {
    GgufBuilder::new()
        .with_string("tokenizer.ggml.model", "gpt2")
        .with_string("tokenizer.ggml.pre", "gpt-2")
        .with_string_array(
            "tokenizer.ggml.tokens",
            &["<unk>", "<s>", "</s>", "a", "b", "c", "ab", "abc", "1", "2"],
        )
        .with_string_array("tokenizer.ggml.merges", &["a b", "ab c"])
        .with_u32("tokenizer.ggml.unknown_token_id", 0)
        .with_u32("tokenizer.ggml.bos_token_id", 1)
        .with_u32("tokenizer.ggml.eos_token_id", 2)
        .with_bool("tokenizer.ggml.add_bos_token", false)
        .with_bool("tokenizer.ggml.add_eos_token", false)
        .build()
}

/// A multi-pattern BPE fixture (StarCoder style). StarCoder applies two
/// sequential patterns: the first (`\p{N}`) splits digits individually, the
/// second handles letters/whitespace. This locks in multi-pattern splitting
/// behaviour without needing a real model.
///
/// Vocabulary (id: piece):
/// - 0: `<unk>` 1: `<s>` 2: `</s>` 3: `1` 4: `2` 5: `a` 6: `b` 7: `ab`
///
/// Merges: `a b` -> `ab`.
///
/// Expected encodes (add_special = false):
/// - `"12"`  -> `[3,4]`   (digits split individually by first pattern)
/// - `"ab"`  -> `[7]`     (letters kept together, then merged)
/// - `"1a"`  -> `[3,5]`
#[must_use]
pub fn bpe_starcoder_fixture() -> Vec<u8> {
    GgufBuilder::new()
        .with_string("tokenizer.ggml.model", "gpt2")
        .with_string("tokenizer.ggml.pre", "starcoder")
        .with_string_array(
            "tokenizer.ggml.tokens",
            &["<unk>", "<s>", "</s>", "1", "2", "a", "b", "ab"],
        )
        .with_string_array("tokenizer.ggml.merges", &["a b"])
        .with_u32("tokenizer.ggml.unknown_token_id", 0)
        .with_u32("tokenizer.ggml.bos_token_id", 1)
        .with_u32("tokenizer.ggml.eos_token_id", 2)
        .with_bool("tokenizer.ggml.add_bos_token", false)
        .with_bool("tokenizer.ggml.add_eos_token", false)
        .build()
}
