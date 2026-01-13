# Tokenizer Compare: llama.cpp vs shimmytok (libshimmy context)

This note answers: “what is the full scope of an ‘adequate’ tokenizer?” and “what gaps exist vs llama.cpp?”.

## 1) What llama.cpp’s tokenizer actually is (surface area)

llama.cpp’s tokenizer is encapsulated in `llama_vocab` and exposed via:

- `llama_tokenize(vocab, text, add_special, parse_special)`
- `llama_token_to_piece(vocab, token, lstrip, special)`
- `llama_detokenize(vocab, tokens, remove_special, unparse_special)`

Internally, llama.cpp selects an algorithm based on `llama_vocab_type`:

- `SPM` (SentencePiece-like, *byte-level*, with byte fallback)
- `BPE` (GPT-2 byte-level BPE)
- `WPM` (WordPiece)
- `UGM` (Unigram / T5-style)
- `RWKV` (greedy)
- `PLAMO2` (Aho-Corasick + DP)

**Important:** Tokenization in llama.cpp is not just “encode/decode”; it includes policy flags and special-token parsing behavior that affect correctness.

## 2) shimmytok’s current design (as vendored in this repo)

shimmytok (v0.5.0) loads vocab/config from GGUF (`tokenizer.ggml.*`) and chooses implementation by `tokenizer.ggml.model` string:

- `"llama" | "mistral" | "gemma"` → `SentencePieceTokenizer`
- `"gpt2" | "qwen" | "qwen2"` → `BPETokenizer`

API surface:

- `Tokenizer::from_gguf_file(path)`
- `encode(text, add_special_tokens: bool) -> Vec<u32>`
- `decode(tokens, skip_special_tokens: bool) -> String`
- `decode_single(token, skip_special_tokens)`
- `token_to_piece(token) -> String` (raw vocab entry)

This is intentionally smaller than llama.cpp’s API.

## 3) What is already “proven good” for TinyLlama Q4_0

In libshimmy we now have oracle-backed tests proving:

- shimmytok `encode(prompt, true)` matches the llama.cpp oracle `prompt_tokens` for at least:
  - hello_world
  - pangram_long

That’s strong evidence that *encoding* for TinyLlama’s tokenizer path is correct.

## 4) Gaps vs llama.cpp (feature diff)

### A) Algorithm coverage gaps (major)

llama.cpp supports many vocab types; shimmytok currently covers only:

- ✅ SPM
- ✅ BPE
- ❌ WPM
- ❌ UGM
- ❌ RWKV
- ❌ PLAMO2

If your goal is “full stack I own”, you either:

- accept a model-support boundary (SPM+BPE only), or
- implement the remaining vocab types and dispatch by actual GGUF vocab type/metadata.

### B) Special-token parsing (major)

llama.cpp has `parse_special` in `llama_tokenize()`:

- When `parse_special=true`, it will detect control/special token strings embedded in the input and emit those tokens as tokens (without doing the usual leading-space behavior).

shimmytok currently has **no equivalent**. It always treats input as plain text.

If you want parity with llama.cpp CLI behavior (people paste `<|eot_id|>` etc), you need this.

### C) Space-prefix semantics are hard-coded in shimmytok SPM (major)

llama.cpp SPM behavior:

- It only prefixes a leading space when `add_space_prefix && previous_fragment_was_special`.

shimmytok SPM behavior today:

- It always prefixes with `▁` if the text doesn’t start with a space.
- It does not consult `tokenizer.ggml.add_space_prefix` despite reading it.

This can cause subtle mismatches on some models / prompt fragment boundaries.

### D) Correct detokenization for SPM “byte tokens” (major)

llama.cpp `token_to_piece()` for SPM/UGM/WPM:

- unescapes whitespace (`▁` → space)
- and if token attr is BYTE, it emits the actual byte (not the textual token name)

shimmytok SPM `decode()` currently:

- concatenates raw token texts and replaces `▁` with space
- **does not** decode byte tokens like `<0x0A>` into `\n`

This is a real correctness hole for “model output to user-visible string” in edge cases.

### E) Token attributes / filtering semantics (medium)

llama.cpp has token attributes (NORMAL/CONTROL/UNKNOWN/BYTE/etc) and:

- `token_to_piece(..., special=false)` suppresses special/control pieces
- `detokenize(..., remove_special, unparse_special)` is more nuanced than “skip_special_tokens”

shimmytok only has:

- `skip_special_tokens` boolean (based on GGUF token types + a few IDs)
- `token_to_piece()` returns raw token text, no `lstrip`, no special suppression.

### F) Additional special token IDs (medium)

llama.cpp exposes many special tokens:

- `eot`, `sep`, `nl`, `mask`, plus FIM tokens (`fim_pre`, `fim_mid`, ...)

shimmytok’s vocab loader only tracks:

- `bos`, `eos`, `unk`, `pad`

For chat models and some sampling/stop criteria, having `eot`/`eog` matters.

### G) Cleanup/normalization flags (model-dependent, medium)

llama.cpp vocab includes flags such as:

- `clean_spaces`, `remove_extra_whitespaces`, `escape_whitespaces`, `treat_whitespace_as_suffix`, `ignore_merges`

shimmytok does not implement these policies.

Some models rely on them; for TinyLlama you may not care, but for “own the full stack”, you will.

## 5) What to implement next (if you want llama.cpp-level parity)

**Tier 1 (must for robust parity):**

1. SPM detokenization that matches llama.cpp `token_to_piece()` (whitespace + byte tokens)
2. Support for `parse_special` (tokenize embedded special/control tokens)
3. Honor `tokenizer.ggml.add_space_prefix` for SPM (and fragment-boundary behavior)
4. Track additional special token IDs (`eot`, `sep`, `nl`, `fim_*`, `mask`) from GGUF and expose them

**Tier 2 (model expansion):**

5. Implement remaining vocab types (WPM/UGM/RWKV/PLAMO2) or explicitly fail-closed with a diagnostic

**Tier 3 (proof):**

6. Differential test harness: compare against llama.cpp `llama_tokenize` / `llama_detokenize` on a corpus

## 6) Practical recommendation for your “owned stack” goal

If you want to keep shimmytok small and dependable:

- Keep the public API minimal
- Internally add an “llama.cpp compatibility mode” that exposes parity toggles:
  - `add_special`
  - `parse_special`
  - `remove_special`
  - `unparse_special`
  - `lstrip`

…and prove correctness by continuously diffing against llama.cpp for each supported vocab type.
