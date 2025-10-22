//! BPE (Byte Pair Encoding) tokenizer implementation
//! Based on GPT-2 style BPE with direct merge rules from GGUF

use crate::vocab::Vocabulary;
use crate::TokenId;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

// Pre-tokenization regex patterns from llama.cpp
// Reference: https://github.com/ggerganov/llama.cpp/blob/master/common/common.cpp

/// GPT-2 pattern (default BPE)
const GPT2_PATTERN: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+";

/// Llama-3 BPE pattern
const LLAMA3_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// DeepSeek LLM pattern
const DEEPSEEK_LLM_PATTERN: &str = r"[\r\n]+|[\p{P}\p{S}]|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+";

/// DeepSeek Coder pattern  
const DEEPSEEK_CODER_PATTERN: &str = r"[\r\n]+|[\p{P}\p{S}\$]|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+";

/// Falcon pattern
const FALCON_PATTERN: &str = r"\n| ?[\p{L}\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+";

/// MPT pattern
/// MPT pattern - Uses GPT-2 pattern (see get_pattern mapping)

/// Falcon pattern

/// StarCoder pattern (also used by Refact, Command-R, SmolLM, Codeshell, Exaone, Minerva)
const STARCODER_PATTERN: &str = r"\p{N}|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)";

/// Bloom pattern (also used by Poro, GPT3-Finnish)
const BLOOM_PATTERN: &str = r"\s+|\S+";

/// Qwen2 pattern
const QWEN2_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// ChatGLM-4 pattern (glm4, chatglm-bpe)
const CHATGLM4_PATTERN: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// DBRX pattern - Same as Llama-3
const DBRX_PATTERN: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Smaug pattern - Same as Llama-3
const SMAUG_PATTERN: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Poro pattern (Finnish-focused) - Same as Bloom
const PORO_PATTERN: &str = r" ?[^(\s|.,!?…。，、।۔،)]+";

/// DeepSeek-v3 pattern (NEW in 2025) 
const DEEPSEEK_V3_PATTERN: &str = r"\p{N}{1,3}|[一-龥぀-ゟ゠-ヿ]+|[!#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Hunyuan-Dense pattern
const HUNYUAN_DENSE_PATTERN: &str = r"\p{N}{1,3}|[一-龥぀-ゟ゠-ヿ]+|[!#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// StableLM2 pattern - Same as Qwen2
const STABLELM2_PATTERN: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Hunyuan pattern - Same as Qwen2
const HUNYUAN_PATTERN: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Viking pattern (Norwegian)
const VIKING_PATTERN: &str = r" ?[^(\s|.,!?…。，、।۔،)]+";

/// GPT3-Finnish pattern - Same as Bloom
const GPT3_FINNISH_PATTERN: &str = r" ?[^(\s|.,!?…。，、।۔،)]+";

/// Refact pattern - Same as StarCoder
const REFACT_PATTERN: &str = r"\p{N}|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)";

/// SmolLM pattern - Same as StarCoder
const SMOLLM_PATTERN: &str = r"\p{N}|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)";

/// Codeshell pattern - Same as StarCoder
const CODESHELL_PATTERN: &str = r"\p{N}|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)";

/// Exaone pattern - Same as StarCoder
const EXAONE_PATTERN: &str = r"\p{N}|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)";

/// Minerva pattern - Same as StarCoder
const MINERVA_PATTERN: &str = r"\p{N}|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)";

/// Trillion pattern - Same as GPT-2
const TRILLION_PATTERN: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)";

/// Granite-Docling pattern - Same as GPT-2
const GRANITE_DOCLING_PATTERN: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)";

/// Tekken pattern (complex case-based tokenization)
const TEKKEN_PATTERN: &str = r"[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))*((?=[\p{L}])([^A-Z]))+|[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))+((?=[\p{L}])([^A-Z]))*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Chameleon pattern (multi-modal with image tokens)
const CHAMELEON_PATTERN: &str = r"<sentinel:[0-9]+>|(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z|([\t\n]|    |  )|\p{N}|[\p{P}!-/:-@\[-`{-~]|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)";

/// GPT-4o pattern (OpenAI)
const GPT4O_PATTERN: &str = r"[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))*((?=[\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))+((?=[\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// KIMI-K2 pattern (Han character handling)
const KIMI_K2_PATTERN: &str = r"\p{Han}+";

/// SuperBPE pattern (number formatting)
const SUPERBPE_PATTERN: &str = r"\p{N}+|(?=(\d{3})+(?!\d))";

/// BailingMoe pattern
const BAILINGMOE_PATTERN: &str = r"'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

/// Seed-Coder pattern
const SEED_CODER_PATTERN: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1}| ?[^\s\p{L}\p{N}\r\n]+|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Grok-2 pattern (xAI)
const GROK_2_PATTERN: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Symbol representing a text fragment during BPE merging
#[derive(Debug, Clone)]
struct Symbol {
    /// Start byte offset in original text
    text_start: usize,
    /// Byte length of this symbol
    text_len: usize,
    /// Index of previous symbol (linked list)
    prev: Option<usize>,
    /// Index of next symbol (linked list)
    next: Option<usize>,
}

/// Bigram candidate for merging, with priority rank
#[derive(Debug, Clone, Eq, PartialEq)]
struct Bigram {
    left: usize,
    right: usize,
    rank: usize,
    text: String,
}

impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> Ordering {
        // Lower rank = higher priority, so reverse the comparison
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct BPETokenizer {
    // Lazily compiled regex patterns
    regex_cache: std::sync::Mutex<HashMap<String, regex::Regex>>,
}

impl Default for BPETokenizer {
    fn default() -> Self {
        BPETokenizer {
            regex_cache: std::sync::Mutex::new(HashMap::new()),
        }
    }
}

impl BPETokenizer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the appropriate regex pattern for a pre-tokenizer type
    fn get_pattern(pre_type: &str) -> &'static str {
        match pre_type {
            // Llama-3 family
            "llama3" | "llama-bpe" | "llama-v3" | "falcon3" => LLAMA3_PATTERN,
            
            // DeepSeek family
            "deepseek-llm" => DEEPSEEK_LLM_PATTERN,
            "deepseek-coder" => DEEPSEEK_CODER_PATTERN,
            "deepseek-v3" => DEEPSEEK_V3_PATTERN,
            "deepseek-r1-qwen" => QWEN2_PATTERN,
            
            // Falcon
            "falcon" => FALCON_PATTERN,
            
            // StarCoder family (many models use this)
            "starcoder" | "refact" | "command-r" | "smollm" | "codeshell" | "exaone" | "minerva" => STARCODER_PATTERN,
            
            // GPT-2 family (most common, many aliases)
            "gpt-2" | "phi-2" | "jina-es" | "jina-de" | "mpt" | "olmo" | "jais" | "trillion" | "granite-docling" | "exaone4" => GPT2_PATTERN,
            
            // Qwen2 family
            "qwen2" | "stablelm2" | "hunyuan" | "megrez" => QWEN2_PATTERN,
            
            // Bloom family
            "bloom" | "poro-chat" | "gpt3-finnish" => BLOOM_PATTERN,
            
            // ChatGLM
            "chatglm4" | "glm4" | "chatglm-bpe" => CHATGLM4_PATTERN,
            
            // Same-as-Llama-3 patterns
            "dbrx" => DBRX_PATTERN,
            "smaug-bpe" => SMAUG_PATTERN,
            
            // Norwegian
            "viking" => VIKING_PATTERN,
            
            // Advanced/Specialized
            "tekken" => TEKKEN_PATTERN,
            "chameleon" => CHAMELEON_PATTERN,
            "gpt-4o" | "llama4" => GPT4O_PATTERN,
            "kimi-k2" => KIMI_K2_PATTERN,
            "superbpe" => SUPERBPE_PATTERN,
            "bailingmoe" | "bailingmoe2" | "llada-moe" => BAILINGMOE_PATTERN,
            "seed-coder" => SEED_CODER_PATTERN,
            "hunyuan-dense" => HUNYUAN_DENSE_PATTERN,
            "grok-2" => GROK_2_PATTERN,
            
            // Default (GPT-2)
            _ => GPT2_PATTERN,
        }
    }

    /// Get or compile the pre-tokenization regex
    fn get_regex(&self, pre_type: &str) -> Result<regex::Regex, String> {
        let mut cache = self
            .regex_cache
            .lock()
            .map_err(|e| format!("Mutex lock failed: {}", e))?;

        if let Some(regex) = cache.get(pre_type) {
            return Ok(regex.clone());
        }

        let pattern = Self::get_pattern(pre_type);
        let regex = regex::Regex::new(pattern)
            .map_err(|e| format!("Failed to compile regex for '{}': {}", pre_type, e))?;

        cache.insert(pre_type.to_string(), regex.clone());
        Ok(regex)
    }

    /// Pre-tokenize text using regex patterns
    fn pre_tokenize(&self, text: &str, vocab: &Vocabulary) -> Result<Vec<String>, String> {
        let pre_type = vocab.pre_type().unwrap_or("gpt2");
        let regex = self.get_regex(pre_type)?;

        Ok(regex
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect())
    }

    /// Apply BPE to a single text fragment
    /// Direct port of llama.cpp llm_tokenizer_bpe_session::tokenize
    fn bpe_fragment(&self, text: &str, vocab: &Vocabulary) -> Result<Vec<TokenId>, crate::Error> {
        // Text is a single word from regex pre-tokenization
        // llama.cpp initializes with UTF-8 characters as symbols

        // Build merge rank map: (left, right) -> rank
        let merge_ranks: HashMap<(String, String), usize> = vocab
            .get_merges()
            .iter()
            .enumerate()
            .map(|(rank, (l, r))| ((l.clone(), r.clone()), rank))
            .collect();

        // Split into UTF-8 characters as initial symbols
        let char_indices: Vec<(usize, char)> = text.char_indices().collect();
        let mut symbols: Vec<Symbol> = Vec::with_capacity(char_indices.len());

        for (i, (byte_pos, _ch)) in char_indices.iter().enumerate() {
            let next_byte_pos = if i + 1 < char_indices.len() {
                char_indices[i + 1].0
            } else {
                text.len()
            };

            symbols.push(Symbol {
                text_start: *byte_pos,
                text_len: next_byte_pos - byte_pos,
                prev: if i == 0 { None } else { Some(i - 1) },
                next: if i + 1 < char_indices.len() {
                    Some(i + 1)
                } else {
                    None
                },
            });
        }

        if symbols.is_empty() {
            return Ok(Vec::new());
        }

        // Build initial work queue with all adjacent bigrams
        let mut work_queue = BinaryHeap::new();
        for i in 0..symbols.len().saturating_sub(1) {
            if let Some(next) = symbols[i].next {
                self.try_add_bigram(i, next, text, &symbols, &merge_ranks, &mut work_queue);
            }
        }

        // Apply merges in priority order
        while let Some(bigram) = work_queue.pop() {
            let left = bigram.left;
            let right = bigram.right;

            // Validate bigram is still valid
            if symbols[left].text_len == 0
                || symbols[right].text_len == 0
                || symbols[left].next != Some(right)
            {
                continue;
            }

            // CRITICAL: Validate that current symbol texts match the merge rule
            // Symbols may have changed since bigram was added to queue
            let left_text =
                &text[symbols[left].text_start..symbols[left].text_start + symbols[left].text_len];
            let right_text = &text
                [symbols[right].text_start..symbols[right].text_start + symbols[right].text_len];

            if let Some(&expected_rank) =
                merge_ranks.get(&(left_text.to_string(), right_text.to_string()))
            {
                if expected_rank != bigram.rank {
                    continue;
                }
            } else {
                continue;
            }

            // Merge: extend left symbol to include right symbol
            symbols[left].text_len += symbols[right].text_len;
            symbols[right].text_len = 0; // Mark right as deleted

            // Update linked list
            symbols[left].next = symbols[right].next;
            if let Some(next) = symbols[right].next {
                symbols[next].prev = Some(left);
            }

            // Add new potential merges with neighbors
            if let Some(prev) = symbols[left].prev {
                self.try_add_bigram(prev, left, text, &symbols, &merge_ranks, &mut work_queue);
            }
            if let Some(next) = symbols[left].next {
                self.try_add_bigram(left, next, text, &symbols, &merge_ranks, &mut work_queue);
            }
        }

        // Convert symbols to token IDs (llama.cpp lines 1101-1118)
        let mut result = Vec::new();
        for sym in &symbols {
            if sym.text_len > 0 {
                let token_text = &text[sym.text_start..sym.text_start + sym.text_len];
                if let Some(id) = vocab.get_token_id(token_text) {
                    result.push(id);
                } else {
                    // Byte fallback: llama.cpp looks up each byte as a single-char string
                    // NOT using hex format <0xXX> - that's for SentencePiece only
                    for byte_char in token_text.chars() {
                        let byte_str = byte_char.to_string();
                        if let Some(id) = vocab.get_token_id(&byte_str) {
                            result.push(id);
                        } else {
                            // Ultimate fallback - use unk token
                            result.push(vocab.unk_token_id());
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    /// Try to add a bigram to the work queue if it's a valid merge
    fn try_add_bigram(
        &self,
        left: usize,
        right: usize,
        text: &str,
        symbols: &[Symbol],
        merge_ranks: &HashMap<(String, String), usize>,
        work_queue: &mut BinaryHeap<Bigram>,
    ) {
        if symbols[left].text_len == 0 || symbols[right].text_len == 0 {
            return;
        }

        let left_text =
            &text[symbols[left].text_start..symbols[left].text_start + symbols[left].text_len];
        let right_text =
            &text[symbols[right].text_start..symbols[right].text_start + symbols[right].text_len];

        if let Some(&rank) = merge_ranks.get(&(left_text.to_string(), right_text.to_string())) {
            work_queue.push(Bigram {
                left,
                right,
                rank,
                text: format!("{}{}", left_text, right_text),
            });
        }
    }

    /// Encode text to token IDs using BPE
    pub fn encode(&self, text: &str, vocab: &Vocabulary) -> Result<Vec<TokenId>, crate::Error> {
        // Validate input size (Issue #10)
        if text.len() > crate::MAX_INPUT_SIZE {
            return Err(crate::Error::TokenizationFailed(format!(
                "Input text too large: {} bytes (max: {})",
                text.len(),
                crate::MAX_INPUT_SIZE
            )));
        }

        // GPT-2 byte-level BPE: convert text bytes to unicode using GPT-2's byte encoder
        let text_encoded = crate::byte_encoder::encode_bytes(text);

        // Pre-tokenize text into fragments
        let fragments = self.pre_tokenize(&text_encoded, vocab).map_err(|e| {
            crate::Error::TokenizationFailed(format!("Pre-tokenization failed: {}", e))
        })?;

        // Apply BPE to each fragment
        let mut result = Vec::new();
        for fragment in fragments {
            let tokens = self.bpe_fragment(&fragment, vocab)?;
            // Check output size to prevent memory exhaustion
            if result.len() + tokens.len() > crate::MAX_OUTPUT_TOKENS {
                return Err(crate::Error::TokenizationFailed(format!(
                    "Output would exceed max tokens: {} (max: {})",
                    result.len() + tokens.len(),
                    crate::MAX_OUTPUT_TOKENS
                )));
            }
            result.extend(tokens);
        }

        Ok(result)
    }

    /// Decode token IDs back to text
    pub fn decode(&self, tokens: &[TokenId], vocab: &Vocabulary) -> Result<String, crate::Error> {
        // Validate all token IDs exist
        for &id in tokens {
            if vocab.get_token_text(id).is_none() {
                return Err(crate::Error::InvalidToken(format!(
                    "Token ID {} not found in vocabulary",
                    id
                )));
            }
        }

        let byte_encoded_text: String = tokens
            .iter()
            .filter_map(|&id| vocab.get_token_text(id))
            .collect::<Vec<_>>()
            .join("");

        // Convert from GPT-2 byte encoding back to normal UTF-8
        let decoded = crate::byte_encoder::decode_bytes(&byte_encoded_text);

        // Validate final decoded size (Issue R3#8) - decoding can expand
        const MAX_DECODED_SIZE: usize = 100 * 1024 * 1024; // 100MB
        if decoded.len() > MAX_DECODED_SIZE {
            return Err(crate::Error::TokenizationFailed(format!(
                "Final decoded text too large: {} bytes (max: {})",
                decoded.len(),
                MAX_DECODED_SIZE
            )));
        }

        Ok(decoded)
    }
}

impl crate::TokenizerImpl for BPETokenizer {
    fn encode(&self, text: &str, vocab: &Vocabulary) -> Result<Vec<TokenId>, crate::Error> {
        BPETokenizer::encode(self, text, vocab)
    }

    fn decode(&self, tokens: &[TokenId], vocab: &Vocabulary) -> Result<String, crate::Error> {
        BPETokenizer::decode(self, tokens, vocab)
    }
}
