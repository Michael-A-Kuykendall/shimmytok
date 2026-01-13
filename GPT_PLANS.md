Below are **pure-Rust ports** of the four remaining tokenizer paths you listed, structured as **separate modules** and written to match the *actual* llama.cpp behavior in `llama-vocab.cpp`.

Two critical notes up front:

1. **llama.cpp “WPM” is not BERT WordPiece (`##`)**. In the current llama.cpp implementation I pulled, `LLAMA_VOCAB_TYPE_WPM` is *“normalize + whitespace split + phantom space + greedy longest-match”* (no `##` continuation logic). See the `llm_tokenizer_wpm_session::tokenize()` implementation. ([Hugging Face][1])
2. **PLaMo-2 (plamo2)** tokenizer is not present in that same `llama-vocab.cpp` snapshot; the clearest reference implementation available is the official PLaMo-2 tokenizer code (Aho–Corasick-ish table + reverse DP). ([Hugging Face][2])

---

## 1) `wpm.rs` — llama.cpp WPM (phantom-space + greedy longest-match)

```rust
// wpm.rs
//
// Port of llama.cpp "WPM tokenizer" (LLAMA_VOCAB_TYPE_WPM).
// Source reference: llm_tokenizer_wpm_session::tokenize + preprocess in llama-vocab.cpp :contentReference[oaicite:2]{index=2}
//
// Key behaviors:
// - Unicode NFD normalize, lowercase
// - Split into "words" on whitespace
// - Punctuation / certain symbols / Chinese chars become single-char "words"
// - Each word is tokenized by greedy longest-match against vocab, with phantom space prefix U+2581 (▁)
// - If any character position in a word cannot be matched: discard all tokens produced for that word
// - If a word yields no tokens, emit unk

use crate::{Error, TokenType, TokenizerImpl, Vocabulary};

pub struct WpmTokenizer {
    // llama.cpp uses vocab.max_token_len() to cap search. We derive it once.
    max_token_len: usize,
}

impl WpmTokenizer {
    pub fn new(vocab: &Vocabulary) -> Self {
        let mut max_len = 0usize;
        for id in 0..(vocab.n_tokens() as u32) {
            if let Some(t) = vocab.get_token_text(id) {
                max_len = max_len.max(t.len());
            }
        }
        Self { max_token_len: max_len }
    }
}

impl TokenizerImpl for WpmTokenizer {
    fn encode(&self, text: &str, vocab: &Vocabulary) -> Result<Vec<u32>, Error> {
        let words = preprocess_wpm(text);

        let phantom_space = "\u{2581}"; // U+2581 ▁
        let mut out: Vec<u32> = Vec::new();

        for w in words {
            if w.is_empty() {
                continue;
            }

            let word1 = format!("{phantom_space}{w}");
            let bytes = word1.as_bytes();
            let n = bytes.len();
            let checkpoint = out.len();

            let mut i = 0usize;
            while i < n {
                let mut matched = false;

                // llama.cpp upper bound: min(n, i + max_token_len + 1) and then j decrements :contentReference[oaicite:3]{index=3}
                let mut j = (i + self.max_token_len + 1).min(n);
                while j > i {
                    let s = &word1[i..j];
                    if let Some(id) = vocab.get_token_id(s) {
                        out.push(id);
                        matched = true;
                        i = j; // next position
                        break;
                    }
                    j -= 1;
                }

                if !matched {
                    // discard all tokens added for this word
                    out.truncate(checkpoint);
                    break;
                }
            }

            if out.len() == checkpoint {
                out.push(vocab.unk_token_id());
            }
        }

        Ok(out)
    }

    fn decode(&self, tokens: &[u32], vocab: &Vocabulary) -> Result<String, Error> {
        // Standard behavior for decode in this style tokenizer: concatenate token strings.
        // (Exact llama.cpp decode semantics may post-process special tokens elsewhere.)
        let mut s = String::new();
        for &t in tokens {
            if let Some(txt) = vocab.get_token_text(t) {
                s.push_str(txt);
            }
        }
        Ok(s)
    }
}

/// Preprocess per llama.cpp:
/// - NFD normalize
/// - lowercase
/// - split on whitespace
/// - punctuation / symbols / "Chinese char" => single-char word boundaries
///
/// IMPORTANT: llama.cpp uses its own unicode helpers; here we implement a pragmatic approximation.
/// If your crate already has unicode normalize + flags, wire those in for exact parity.
/// :contentReference[oaicite:4]{index=4}
fn preprocess_wpm(text: &str) -> Vec<String> {
    // If you already have a shared unicode module (likely, given SPM/BPE),
    // replace this with the same NFD + lowercase + flag logic used elsewhere.
    //
    // For now: lowercase + simplistic whitespace split, plus isolate ASCII punctuation.
    let lower = text.to_lowercase();

    let mut words: Vec<String> = vec![String::new()];
    for ch in lower.chars() {
        if ch.is_whitespace() {
            if !words.last().unwrap().is_empty() {
                words.push(String::new());
            }
            continue;
        }

        let is_ascii_punct = ch.is_ascii_punctuation();
        if is_ascii_punct {
            if !words.last().unwrap().is_empty() {
                words.push(String::new());
            }
            words.last_mut().unwrap().push(ch);
            words.push(String::new());
        } else {
            words.last_mut().unwrap().push(ch);
        }
    }

    if words.last().map(|w| w.is_empty()).unwrap_or(false) {
        words.pop();
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wpm_basic_smoke() {
        // This is a structural smoke test; exact IDs depend on your vocab.
        // You should replace with a GGUF-derived vocab fixture and assert exact token IDs.
        let _ = preprocess_wpm("Hello, world!");
    }
}
```

---

## 2) `ugm.rs` — llama.cpp UGM (unigram Viterbi-like over a trie)

This is the **optimized Viterbi unigram** described in the source comments and code: advance by UTF-8 codepoint; at each offset traverse a trie of possible tokens; keep best score to each end position; if no token covers the *whole* codepoint, allow `<unk>` with penalty; then backtrack. ([Hugging Face][3])

```rust
// ugm.rs
//
// Port of llama.cpp UGM tokenizer session (unigram / SentencePiece-style Viterbi).
// Source comment + core loops in llm_tokenizer_ugm_session::tokenize :contentReference[oaicite:6]{index=6}
//
// Notes:
// - llama.cpp uses a "naive_trie" that supports traverse(byte) and terminal node values
// - Scores: normal tokens use token_data.score; user_defined tokens get score 0.0 :contentReference[oaicite:7]{index=7}
// - unknown_token_score = min_score - penalty (penalty defaults 10.0) :contentReference[oaicite:8]{index=8}
//
// This module assumes you can build a trie over UTF-8 bytes of token strings.

use crate::{Error, TokenType, TokenizerImpl, Vocabulary};

#[derive(Clone)]
struct TrieNode {
    next: std::collections::HashMap<u8, usize>,
    value: Option<u32>,
}
#[derive(Clone)]
struct NaiveTrie {
    nodes: Vec<TrieNode>,
}
impl NaiveTrie {
    fn new() -> Self {
        Self { nodes: vec![TrieNode { next: Default::default(), value: None }] }
    }
    fn insert(&mut self, s: &str, id: Option<u32>) {
        let mut cur = 0usize;
        for &b in s.as_bytes() {
            let nxt = *self.nodes[cur].next.entry(b).or_insert_with(|| {
                self.nodes.push(TrieNode { next: Default::default(), value: None });
                self.nodes.len() - 1
            });
            cur = nxt;
        }
        if let Some(id) = id {
            self.nodes[cur].value = Some(id);
        }
    }
    fn traverse(&self, node: usize, b: u8) -> Option<usize> {
        self.nodes[node].next.get(&b).copied()
    }
    fn value(&self, node: usize) -> Option<u32> {
        self.nodes[node].value
    }
}

pub struct UgmTokenizer {
    trie: NaiveTrie,
    user_defined_trie: NaiveTrie,
    unknown_token_score: f64,
}

impl UgmTokenizer {
    pub fn new(vocab: &Vocabulary) -> Self {
        let mut trie = NaiveTrie::new();
        let mut user_defined_trie = NaiveTrie::new();

        let mut min_score = f64::INFINITY;
        let mut _max_score = f64::NEG_INFINITY;

        for id in 0..(vocab.n_tokens() as u32) {
            let txt = match vocab.get_token_text(id) {
                Some(t) => t,
                None => continue,
            };
            let ttype = vocab.get_token_type(id);

            // llama.cpp: insert normals, user_defined, unused into token_matcher :contentReference[oaicite:9]{index=9}
            if matches!(ttype, TokenType::Normal | TokenType::UserDefined | TokenType::Unused) {
                trie.insert(txt, Some(id));
            }
            if matches!(ttype, TokenType::UserDefined) {
                user_defined_trie.insert(txt, Some(id));
            }

            if matches!(ttype, TokenType::Normal) {
                let sc = vocab.get_token_score(id) as f64;
                min_score = min_score.min(sc);
                _max_score = _max_score.max(sc);
            }
        }

        let unknown_token_score_penalty = 10.0f64;
        let unknown_token_score = min_score - unknown_token_score_penalty;

        Self { trie, user_defined_trie, unknown_token_score }
    }
}

impl TokenizerImpl for UgmTokenizer {
    fn encode(&self, text: &str, vocab: &Vocabulary) -> Result<Vec<u32>, Error> {
        let normalized = normalize_ugm(text); // hook your real normalization here (xcda etc.)
        if normalized.is_empty() {
            return Ok(Vec::new());
        }

        // Best tokenization DP table:
        // tokenization_results[pos] = (token_id, start_pos, score_sum)
        // Initialized to -INF score except pos=0 is 0 score :contentReference[oaicite:10]{index=10}
        #[derive(Clone, Copy)]
        struct Best {
            token: u32,
            start: usize,
            score: f64,
        }
        let n = normalized.len();
        let mut best: Vec<Best> = vec![
            Best { token: vocab.unk_token_id(), start: 0, score: f64::NEG_INFINITY };
            n + 1
        ];
        best[0] = Best { token: vocab.unk_token_id(), start: 0, score: 0.0 };

        let bytes = normalized.as_bytes();
        let mut input_offset = 0usize;

        while input_offset < n {
            // determine size of current UTF-8 codepoint in bytes
            let cp_len = utf8_cp_len(bytes[input_offset]).min(n - input_offset);

            let current_best = best[input_offset];

            // Traverse trie from this position to find all prefix tokens :contentReference[oaicite:11]{index=11}
            let mut prefix_offset = input_offset;
            let mut node_opt = self.trie.traverse(0, bytes[prefix_offset]);
            prefix_offset += 1;

            let mut single_codepoint_token_found = false;

            while prefix_offset <= n {
                let node = match node_opt {
                    Some(x) => x,
                    None => break,
                };

                if let Some(token_id) = self.trie.value(node) {
                    if prefix_offset - input_offset == cp_len {
                        single_codepoint_token_found = true;
                    }

                    let token_score = match vocab.get_token_type(token_id) {
                        TokenType::UserDefined => 0.0, // llama.cpp sets user-defined token score to 0 :contentReference[oaicite:12]{index=12}
                        _ => vocab.get_token_score(token_id) as f64,
                    };

                    let challenger = current_best.score + token_score;
                    if challenger > best[prefix_offset].score {
                        best[prefix_offset] = Best { token: token_id, start: input_offset, score: challenger };
                    }
                }

                if prefix_offset == n {
                    break;
                }
                node_opt = self.trie.traverse(node, bytes[prefix_offset]);
                prefix_offset += 1;
            }

            // Unknown-token path if no token covers whole codepoint :contentReference[oaicite:13]{index=13}
            if !single_codepoint_token_found {
                let next = input_offset + cp_len;
                let challenger = current_best.score + self.unknown_token_score;
                if challenger > best[next].score {
                    best[next] = Best { token: vocab.unk_token_id(), start: input_offset, score: challenger };
                }
            }

            input_offset += cp_len;
        }

        // Backtrack from end to build tokens :contentReference[oaicite:14]{index=14} (backtrack described below that snippet)
        let mut out_rev: Vec<u32> = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let b = best[pos];
            out_rev.push(b.token);
            pos = b.start;
        }
        out_rev.reverse();
        Ok(out_rev)
    }

    fn decode(&self, tokens: &[u32], vocab: &Vocabulary) -> Result<String, Error> {
        let mut s = String::new();
        for &t in tokens {
            if let Some(txt) = vocab.get_token_text(t) {
                s.push_str(txt);
            }
        }
        Ok(s)
    }
}

fn utf8_cp_len(first: u8) -> usize {
    match first {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    }
}

/// Placeholder normalization.
/// llama.cpp does normalization (optionally via precompiled charsmap XCDA) before tokenization :contentReference[oaicite:15]{index=15}.
/// For exact parity, route this through your existing normalization pipeline / GGUF charsmap parsing.
fn normalize_ugm(text: &str) -> String {
    text.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ugm_smoke() {
        let _ = utf8_cp_len(b'a');
    }
}
```

---

## 3) `rwkv.rs` — llama.cpp RWKV (unescape vocab tokens + greedy trie longest match)

Key llama.cpp behaviors:

* Vocab tokens are stored as *escaped strings*; llama.cpp “unescapes” them into raw bytes (`\n`, `\t`, `\r`, `\xNN`, etc.) ([Hugging Face][1])
* Build a trie over those bytes; then tokenize input greedily by **longest match**; if no match at a position, emit `<unk>` and advance one byte. ([Hugging Face][1])

```rust
// rwkv.rs
//
// Port of llama.cpp RWKV tokenizer.
// Source: llama_unescape_rwkv_token + llm_tokenizer_rwkv_session::tokenize :contentReference[oaicite:18]{index=18}

use crate::{Error, TokenizerImpl, Vocabulary};

#[derive(Clone)]
struct TrieNode {
    next: std::collections::HashMap<u8, usize>,
    value: Option<u32>,
}
#[derive(Clone)]
struct NaiveTrie {
    nodes: Vec<TrieNode>,
}
impl NaiveTrie {
    fn new() -> Self {
        Self { nodes: vec![TrieNode { next: Default::default(), value: None }] }
    }
    fn insert_bytes(&mut self, bs: &[u8], id: u32) {
        let mut cur = 0usize;
        for &b in bs {
            let nxt = *self.nodes[cur].next.entry(b).or_insert_with(|| {
                self.nodes.push(TrieNode { next: Default::default(), value: None });
                self.nodes.len() - 1
            });
            cur = nxt;
        }
        self.nodes[cur].value = Some(id);
    }
}

pub struct RwkvTokenizer {
    trie: NaiveTrie,
}

impl RwkvTokenizer {
    pub fn new(vocab: &Vocabulary) -> Self {
        let mut trie = NaiveTrie::new();
        for id in 0..(vocab.n_tokens() as u32) {
            if let Some(txt) = vocab.get_token_text(id) {
                let raw = unescape_rwkv_token(txt);
                trie.insert_bytes(&raw, id);
            }
        }
        Self { trie }
    }
}

impl TokenizerImpl for RwkvTokenizer {
    fn encode(&self, text: &str, vocab: &Vocabulary) -> Result<Vec<u32>, Error> {
        let bs = text.as_bytes();
        let mut out: Vec<u32> = Vec::new();
        let mut pos = 0usize;

        while pos < bs.len() {
            // first byte must exist in trie
            let mut node = match self.trie.nodes[0].next.get(&bs[pos]).copied() {
                Some(n) => n,
                None => {
                    out.push(vocab.unk_token_id());
                    pos += 1;
                    continue;
                }
            };

            let mut best_id: u32 = 0;
            let mut best_end: usize = pos;

            let mut i = pos + 1;
            loop {
                if let Some(v) = self.trie.nodes[node].value {
                    best_id = v;
                    best_end = i;
                }
                if i >= bs.len() {
                    break;
                }
                match self.trie.nodes[node].next.get(&bs[i]).copied() {
                    Some(n) => {
                        node = n;
                        i += 1;
                    }
                    None => break,
                }
            }

            out.push(best_id);
            pos = best_end.max(pos + 1);
        }

        Ok(out)
    }

    fn decode(&self, tokens: &[u32], vocab: &Vocabulary) -> Result<String, Error> {
        let mut s = String::new();
        for &t in tokens {
            if let Some(txt) = vocab.get_token_text(t) {
                s.push_str(txt);
            }
        }
        Ok(s)
    }
}

/// llama.cpp unescape routine port :contentReference[oaicite:19]{index=19}
fn unescape_rwkv_token(escaped: &str) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(escaped.len());
    let mut escaping = false;
    let mut hex_remaining: u8 = 0;
    let mut hex_acc: u8 = 0;

    for &b in escaped.as_bytes() {
        if hex_remaining != 0 {
            let v = if (b as char) >= 'a' { b - b'a' + 10 } else { b - b'0' };
            hex_acc = (hex_acc << 4) + v;
            hex_remaining -= 1;
            if hex_remaining == 0 {
                out.push(hex_acc);
                hex_acc = 0;
            }
            continue;
        }

        if escaping {
            match b {
                b't' => out.push(b'\t'),
                b'n' => out.push(b'\n'),
                b'r' => out.push(b'\r'),
                b'x' => hex_remaining = 2,
                _ => out.push(b),
            }
            escaping = false;
            continue;
        }

        if b == b'\\' {
            escaping = true;
            continue;
        }

        out.push(b);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rwkv_unescape_basic() {
        assert_eq!(unescape_rwkv_token("\\n"), vec![b'\n']);
        assert_eq!(unescape_rwkv_token("\\x41"), vec![b'A']);
        assert_eq!(unescape_rwkv_token("abc"), b"abc".to_vec());
    }
}
```

---

## 4) `plamo2.rs` — PLaMo-2 (table-driven “Aho–Corasick style” + reverse DP + byte fallback)

This is a direct Rust translation of the official tokenizer logic:

* Build a flattened table over suffixes/prefixes with scores scaled by `1e4`
* DP from end to start minimizing a cost (`scores[i + len] - score`)
* If best path uses UNKNOWN sentinel (`token_id = -1`), fallback to UTF-8 byte tokens for that codepoint
* When ties occur, the table ordering (longer piece first) ensures longest-match preference ([Hugging Face][2])

```rust
// plamo2.rs
//
// Port of PLaMo-2 tokenizer core algorithm (table-driven matching + reverse DP + byte fallback).
// Reference: pfnet/plamo-2-translate-eval tokenization_plamo.py :contentReference[oaicite:21]{index=21}
//
// This module is intentionally self-contained so you can swap in your GGUF-provided vocab
// (token strings, scores, types incl BYTE tokens like "<0xNN>").
//
// IMPORTANT: Exact parity depends on:
// - having BYTE tokens present for all 256 bytes (the reference asserts this)
// - scaling/rounding scores exactly as reference: round(score * 1e4)
// - unknown sentinel score constants

use crate::{Error, TokenType, TokenizerImpl, Vocabulary};

const INVALID_SCORE: i32 = -20_000_000;
const UNKNOWN_SCORE: i32 = -10_000_000;

// table columns (match ref impl)
const T_PIECE_LEN: usize = 0;
const T_TOKEN_ID: usize = 1;
const T_SCORE: usize = 2;
const T_PIECE_ID: usize = 3;

// path columns
const P_TOKEN_LEN: usize = 0;
const P_TOKEN_ID: usize = 1;
const P_NUM_TOKENS: usize = 2;

#[derive(Clone)]
pub struct Plamo2Tokenizer {
    // Mapping from byte (0..255) => token_id (for "<0xNN>" byte fallback)
    byte_token: [u32; 256],

    // Mapping "piece code" => suffix_id:
    // piece_code = (cpt_first_char << 32) | suffix_piece_id
    to_suffix_id: std::collections::HashMap<u64, i32>,

    // Flattened table: rows of [piece_len, token_id, score, piece_id]
    table: Vec<[i32; 4]>,
}

impl Plamo2Tokenizer {
    pub fn new(vocab: &Vocabulary) -> Result<Self, Error> {
        // Build vocab list similar to reference: token string + score + type
        // and ensure BYTE tokens exist for every byte.
        let mut byte_token = [0u32; 256];

        let mut suffix_to_score: std::collections::HashMap<String, Option<f64>> = std::collections::HashMap::new();
        let mut token_to_id: std::collections::HashMap<String, u32> = std::collections::HashMap::new();

        for id in 0..(vocab.n_tokens() as u32) {
            let tok = match vocab.get_token_text(id) {
                Some(t) => t.to_string(),
                None => continue,
            };
            token_to_id.insert(tok.clone(), id);

            if matches!(vocab.get_token_type(id), TokenType::Byte) {
                // Expect "<0xNN>"
                if tok.len() == 6 && tok.starts_with("<0x") && tok.ends_with('>') {
                    let hex = &tok[3..5];
                    if let Ok(b) = u8::from_str_radix(hex, 16) {
                        byte_token[b as usize] = id;
                    }
                }
                continue;
            }

            let sc = vocab.get_token_score(id) as f64;
            suffix_to_score.insert(tok.clone(), Some(sc));

            // ensure all suffixes exist (with None meaning "not a valid token") :contentReference[oaicite:22]{index=22}
            let chars: Vec<char> = tok.chars().collect();
            for i in 1..chars.len() {
                let suf: String = chars[i..].iter().collect();
                suffix_to_score.entry(suf).or_insert(None);
            }
        }

        // Basic validation: all byte tokens must be set (ref asserts this) :contentReference[oaicite:23]{index=23}
        for i in 0..256 {
            if byte_token[i] == 0 {
                return Err(Error::msg("PLaMo-2: missing BYTE token(s) <0xNN> in vocab"));
            }
        }

        // Collect suffixes + "" and sort by reversed string
        let mut suffixes: Vec<String> = suffix_to_score.keys().cloned().collect();
        suffixes.push(String::new());
        suffixes.sort_by(|a, b| a.chars().rev().collect::<String>().cmp(&b.chars().rev().collect::<String>()));

        // Build suffix_to_id (piece_id base) and to_suffix_id mapping :contentReference[oaicite:24]{index=24}
        let mut suffix_to_id: std::collections::HashMap<String, i32> = std::collections::HashMap::new();
        let mut to_suffix_id: std::collections::HashMap<u64, i32> = std::collections::HashMap::new();

        let mut num_pieces: i32 = 0;
        for s in &suffixes {
            suffix_to_id.insert(s.clone(), num_pieces);

            if !s.is_empty() {
                let mut it = s.chars();
                let first = it.next().unwrap() as u32;
                let rest: String = it.collect();
                let rest_id = *suffix_to_id.get(&rest).unwrap_or(&0);
                let code = ((first as u64) << 32) | (rest_id as u32 as u64);
                to_suffix_id.insert(code, num_pieces);
            }

            // 1 (sentinel) + count of prefixes that exist in suffix_to_score (including “invalid” None entries)
            let mut prefixes = 0i32;
            let chars: Vec<char> = s.chars().collect();
            for i in 1..=chars.len() {
                let p: String = chars[..i].iter().collect();
                if suffix_to_score.contains_key(&p) {
                    prefixes += 1;
                }
            }
            num_pieces += 1 + prefixes;
        }

        // Build flattened table, pieces in decreasing length, then sentinel row :contentReference[oaicite:25]{index=25}
        let mut table: Vec<[i32; 4]> = Vec::with_capacity(num_pieces as usize);
        for suffix in &suffixes {
            let chars: Vec<char> = suffix.chars().collect();

            for piece_len in (1..=chars.len()).rev() {
                let piece: String = chars[..piece_len].iter().collect();
                let score_opt = suffix_to_score.get(&piece).cloned();
                if score_opt.is_none() {
                    continue;
                }

                let token_id = token_to_id.get(&piece).copied().map(|x| x as i32).unwrap_or(-1);
                let score_i32 = match score_opt.unwrap() {
                    Some(sc) => (sc * 1e4).round() as i32, // ref: round(score * 1e4) :contentReference[oaicite:26]{index=26}
                    None => INVALID_SCORE,
                };
                let piece_id = *suffix_to_id.get(&piece).unwrap_or(&0);

                table.push([piece_len as i32, token_id, score_i32, piece_id]);
            }

            // sentinel row: len=1, token_id=-1, score=UNKNOWN_SCORE :contentReference[oaicite:27]{index=27}
            table.push([1, -1, UNKNOWN_SCORE, 0]);
        }

        Ok(Self { byte_token, to_suffix_id, table })
    }
}

impl TokenizerImpl for Plamo2Tokenizer {
    fn encode(&self, text: &str, _vocab: &Vocabulary) -> Result<Vec<u32>, Error> {
        // Convert to Unicode scalar values (code points)
        let data: Vec<u32> = text.chars().map(|c| c as u32).collect();
        let n = data.len();

        // DP arrays
        let mut scores: Vec<i64> = vec![i64::MAX / 4; n + 1];
        scores[n] = 0;

        let mut path: Vec<[i32; 3]> = vec![[0, 0, 0]; n + 1];

        let mut suffix_id: i32 = 0;

        for i in (0..n).rev() {
            let c = data[i] as u64;

            // Find next suffix_id by iterating suffix prefixes until found or sentinel :contentReference[oaicite:28]{index=28}
            let mut p = suffix_id as usize;
            while p < self.table.len() {
                let piece_id = self.table[p][T_PIECE_ID] as u32 as u64;
                let code = (c << 32) | piece_id;
                suffix_id = *self.to_suffix_id.get(&code).unwrap_or(&0);

                let score_here = self.table[p][T_SCORE];
                if suffix_id > 0 || score_here == UNKNOWN_SCORE {
                    break;
                }
                p += 1;
            }

            // Evaluate candidates from suffix_id forward; break at sentinel :contentReference[oaicite:29]{index=29}
            let mut p2 = suffix_id as usize;
            while p2 < self.table.len() {
                let score_i32 = self.table[p2][T_SCORE];
                if score_i32 > INVALID_SCORE {
                    let piece_len = self.table[p2][T_PIECE_LEN] as usize;
                    if i + piece_len <= n {
                        let s = scores[i + piece_len] - (score_i32 as i64);
                        if s < scores[i] {
                            scores[i] = s;
                            path[i][P_TOKEN_LEN] = piece_len as i32;
                            path[i][P_TOKEN_ID] = self.table[p2][T_TOKEN_ID];
                            path[i][P_NUM_TOKENS] = path[i + piece_len][P_NUM_TOKENS] + 1;

                            if score_i32 == UNKNOWN_SCORE {
                                // add bytes needed for UTF-8 encoding minus 1 (ref behavior) :contentReference[oaicite:30]{index=30}
                                let c32 = data[i];
                                path[i][P_NUM_TOKENS] += (c32 >= 0x80) as i32
                                    + (c32 >= 0x800) as i32
                                    + (c32 >= 0x10000) as i32;
                            }
                        }
                    }
                }

                if score_i32 == UNKNOWN_SCORE {
                    break;
                }
                p2 += 1;
            }

            if path[i][P_TOKEN_LEN] <= 0 {
                // Fail-closed: always progress at least 1
                path[i][P_TOKEN_LEN] = 1;
                path[i][P_TOKEN_ID] = -1;
                path[i][P_NUM_TOKENS] = path[i + 1][P_NUM_TOKENS] + 1;
            }
        }

        // Decode best path forward :contentReference[oaicite:31]{index=31}
        let mut out: Vec<u32> = Vec::with_capacity(path[0][P_NUM_TOKENS] as usize);
        let mut pos = 0usize;

        while pos < n {
            let token_id = path[pos][P_TOKEN_ID];
            if token_id >= 0 {
                out.push(token_id as u32);
            } else {
                // byte fallback: encode this codepoint to UTF-8 bytes and map each byte :contentReference[oaicite:32]{index=32}
                let ch = std::char::from_u32(data[pos]).unwrap_or('\u{FFFD}');
                let mut buf = [0u8; 4];
                let s = ch.encode_utf8(&mut buf).len();
                for &b in &buf[..s] {
                    out.push(self.byte_token[b as usize]);
                }
            }

            let adv = path[pos][P_TOKEN_LEN] as usize;
            pos += adv.max(1);
        }

        Ok(out)
    }

    fn decode(&self, _tokens: &[u32], _vocab: &Vocabulary) -> Result<String, Error> {
        // For PLaMo-2, decode semantics usually depend on special handling of byte tokens.
        // Implementing exact decode requires mapping BYTE token IDs back to bytes and stitching.
        Err(Error::msg("PLaMo-2 decode not implemented (encode parity prioritized)"))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn plamo2_smoke() {
        // Real parity tests require an actual PLaMo-2 tokenizer.jsonl-derived GGUF vocab fixture.
        assert_eq!("a".chars().count(), 1);
    }
}
```

---

# Risk/Assumption Audit

* **Assumption:** Your earlier description of WPM as “BERT WordPiece with `##`” is not what llama.cpp’s `LLAMA_VOCAB_TYPE_WPM` does in the implementation I retrieved. If you truly need **BERT WordPiece**, you likely need a *different* vocab type / pre-tokenizer path than llama.cpp’s “WPM”. ([Hugging Face][1])
* **Assumption:** Your crate already has (or will accept adding) the **same Unicode normalization/flag logic** used by your existing SPM/BPE paths; my WPM/UGM normalization stubs will not guarantee byte-for-byte parity until wired into your exact unicode routines. ([Hugging Face][1])
* **Assumption:** PLaMo-2 parity depends on having **all 256 `<0xNN>` byte tokens** present and correctly typed in the GGUF-derived vocab; otherwise byte fallback cannot match the reference behavior. ([Hugging Face][2])
* **Risk:** `Plamo2Tokenizer::decode()` is intentionally not implemented here; if you need round-trip decode parity, you must define the canonical reconstruction rules for BYTE + normal tokens for your product (and match whichever reference you consider source-of-truth).

# Action Items for Verification

1. **Generate golden test vectors**: for each vocab type (WPM/UGM/RWKV/PLAMO2), run llama.cpp tokenization on a fixed set of prompts and persist `(prompt, token_ids)` as fixtures.
2. **Wire normalization exactly**:

   * For **WPM**, implement the exact NFD + punctuation/symbol/chinese splitting rules (do not rely on simplistic ASCII checks). ([Hugging Face][1])
   * For **UGM**, implement the charsmap/XCDA normalization if your GGUF provides it; otherwise confirm parity expectations with “no charsmap” models. ([Hugging Face][1])
3. **RWKV**: verify your GGUF loader preserves the escaped token strings exactly (backslashes, hex digits), then confirm `unescape_rwkv_token()` reproduces llama.cpp outputs for known escaped tokens. ([Hugging Face][1])
4. **PLaMo-2**: validate score scaling: `round(score * 1e4)` and the sentinel constants match the official tokenizer. ([Hugging Face][2])

# Flagged Claims Section

* “These match llama.cpp exactly” is **not proven** until you run fixture comparisons against the same llama.cpp commit/model artifacts (especially for Unicode normalization).
* “WPM is WordPiece with `##`” is **contradicted** by the llama.cpp WPM implementation source I retrieved. ([Hugging Face][1])
* “plamo2 lives in llama-vocab.cpp” is **not supported** by the `llama-vocab.cpp` snapshot I used; the PLaMo-2 tokenizer reference here comes from the official PLaMo repository code. ([Hugging Face][2])

If you paste (or point me to) your existing `TokenizerImpl` error type + `TokenType` enum + any shared unicode utilities you already have in `shimmytok`, I can tighten the remaining normalization stubs into your house style without changing the algorithmic core.

[1]: https://huggingface.co/johnbenac/sesame-csm-1b-GGUF-encoder/blob/374c7f1654eea5a5ab786cf58abc7a08e56d213e/src/llama-vocab.cpp "src/llama-vocab.cpp · johnbenac/sesame-csm-1b-GGUF-encoder at 374c7f1654eea5a5ab786cf58abc7a08e56d213e"
[2]: https://huggingface.co/pfnet/plamo-2-translate-eval/blob/main/tokenization_plamo.py "tokenization_plamo.py · pfnet/plamo-2-translate-eval at main"
[3]: https://huggingface.co/spaces/natasa365/whisper.cpp/blob/d7d82b9b2fff88707fd8c757135f395ad295fe20/examples/talk-llama/llama-vocab.cpp "examples/talk-llama/llama-vocab.cpp · natasa365/whisper.cpp at d7d82b9b2fff88707fd8c757135f395ad295fe20"
