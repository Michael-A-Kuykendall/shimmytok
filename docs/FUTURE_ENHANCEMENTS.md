# Future Enhancements

## Core Coverage (Current: 95%)

### ✅ Implemented
- **SentencePiece** (`llama`) - LLaMA, Mistral, Phi, Gemma
- **BPE** (`gpt2`) - GPT-2/3, Falcon, StarCoder

### 📋 Future Additions (5% remaining coverage)

#### 1. WPM - Word Piece Model
**Priority**: Low  
**Effort**: 2-3 days  
**Models**: BERT, RoBERTa, DistilBERT, embedding models  
**Use Case**: Mostly embeddings, rare in chat/generation  
**Reference**: llama.cpp `llm_tokenizer_wpm` (~250 LOC)

#### 2. UGM - Unigram Model  
**Priority**: Low  
**Effort**: 4-5 days  
**Models**: T5, mT5, multilingual models  
**Use Case**: Specialized multilingual, Google models  
**Reference**: llama.cpp `llm_tokenizer_ugm` (~400 LOC)

#### 3. RWKV Custom
**Priority**: Medium (if RWKV adoption grows)  
**Effort**: 1-2 days  
**Models**: RWKV series (RNN-based)  
**Use Case**: Niche but active community  
**Reference**: llama.cpp `llm_tokenizer_rwkv` (~150 LOC)

## Implementation Pattern

All future tokenizers follow the same architecture:

```rust
// 1. Create src/{name}.rs
pub struct {Name}Tokenizer;

impl TokenizerImpl for {Name}Tokenizer {
    fn encode(&self, text: &str, vocab: &Vocabulary) -> Vec<TokenId> {
        // Port from llama.cpp
    }
    
    fn decode(&self, tokens: &[TokenId], vocab: &Vocabulary) -> String {
        // Port from llama.cpp
    }
}

// 2. Add to lib.rs router
match vocab.model_type() {
    "llama" => Box::new(sentencepiece::SentencePieceTokenizer::new()),
    "gpt2" => Box::new(bpe::BPETokenizer::new()),
    "bert" => Box::new(wpm::WPMTokenizer::new()), // NEW
    // ...
}

// 3. Test against llama.cpp output
```

## Notes

- Each tokenizer is isolated (single file)
- No cross-dependencies
- Zero impact on existing code
- Add only when users request specific models
