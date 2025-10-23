# BPE Pattern Validation Plan

**System Specs:**
- GPU: NVIDIA GeForce RTX 3060 (12GB VRAM)
- Available Disk: 13GB
- Strategy: One model at a time, download → test → delete

## Phase 1: Core Patterns (MUST DO)

### 1. GPT-2 Pattern ✅ READY
- **Pattern**: `gpt2` (default)
- **Model**: `gpt2.Q4_K_M.gguf` 
- **Size**: ~260MB
- **Status**: Already downloaded
- **Also validates**: phi-2, jina-es, jina-de, mpt, olmo, jais, trillion, granite-docling
- **Download**: N/A (have it)

### 2. Qwen2 Pattern ⏳ PENDING
- **Pattern**: `qwen2`
- **Model**: `Qwen2-7B-Instruct-Q4_K_M.gguf`
- **Size**: ~4.4GB
- **Status**: Not downloaded
- **Also validates**: stablelm2, hunyuan, megrez, deepseek-r1-qwen
- **Download**: https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF

### 3. StarCoder Pattern ⏳ PENDING
- **Pattern**: `starcoder`
- **Model**: `starcoder-3b.Q4_K_M.gguf`
- **Size**: ~2GB
- **Status**: Not downloaded
- **Also validates**: refact, command-r, smollm, codeshell, exaone, minerva
- **Download**: https://huggingface.co/bigcode/starcoder-3b-GGUF

### 4. Llama-3 BPE Pattern ⏳ PENDING
- **Pattern**: `llama3` / `llama-bpe`
- **Model**: `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` (BPE variant, NOT SentencePiece)
- **Size**: ~4.9GB
- **Status**: Not downloaded
- **Also validates**: falcon3, dbrx, smaug-bpe
- **Download**: Need to find BPE variant (not the SentencePiece one we have)

## Phase 2: Important Secondary

### 5. DeepSeek-Coder Pattern ⏳ PENDING
- **Pattern**: `deepseek-coder`
- **Model**: `deepseek-coder-6.7b-instruct.Q4_K_M.gguf`
- **Size**: ~4GB
- **Status**: Not downloaded
- **Download**: https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct-GGUF

### 6. DeepSeek-LLM Pattern ⏳ PENDING
- **Pattern**: `deepseek-llm`
- **Model**: `deepseek-llm-7b-chat.Q4_K_M.gguf`
- **Size**: ~4.4GB
- **Status**: Not downloaded
- **Download**: https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat-GGUF

### 7. BLOOM Pattern ⏳ PENDING
- **Pattern**: `bloom`
- **Model**: `bloomz-7b1.Q4_K_M.gguf`
- **Size**: ~4.4GB
- **Status**: Not downloaded
- **Also validates**: poro-chat, gpt3-finnish
- **Download**: https://huggingface.co/bigscience/bloomz-7b1-GGUF

## Phase 3: If Time/Space Permits

### 8. Falcon Pattern ⏳ PENDING
- **Pattern**: `falcon`
- **Model**: `falcon-7b.Q4_K_M.gguf`
- **Size**: ~4.2GB
- **Status**: Not downloaded
- **Download**: https://huggingface.co/tiiuae/falcon-7b-GGUF

### 9. ChatGLM-4 Pattern ⏳ PENDING
- **Pattern**: `chatglm4` / `glm4`
- **Model**: `glm-4-9b-chat.Q4_K_M.gguf`
- **Size**: ~5.5GB (might be tight)
- **Status**: Not downloaded
- **Download**: https://huggingface.co/THUDM/glm-4-9b-chat-GGUF

### 10. Phi-2 Pattern (GPT-2 validation) ⏳ PENDING
- **Pattern**: `phi-2` (uses GPT-2 pattern)
- **Model**: `phi-2.Q4_K_M.gguf`
- **Size**: ~1.6GB
- **Status**: Not downloaded
- **Download**: https://huggingface.co/microsoft/phi-2-GGUF

## Patterns We CANNOT Test Locally

**Too Large (>12GB VRAM even at Q4):**
- DeepSeek-v3 (685B) - requires 37GB+
- GPT-4o - no GGUF available
- Grok-2 (314B) - requires 17GB+
- Hunyuan-Dense - requires 13B+ model
- Most advanced 2025 models

**Strategy**: Mark as "Implemented but untested" until users with bigger hardware validate

## Download Order

1. ✅ GPT-2 (have it)
2. ⏳ Qwen2-7B (~4.4GB)
3. ⏳ StarCoder-3B (~2GB)
4. ⏳ Llama-3-8B BPE (~4.9GB) - need to find correct variant
5. ⏳ DeepSeek-Coder-6.7B (~4GB)
6. ⏳ DeepSeek-LLM-7B (~4.4GB)
7. ⏳ BLOOM-7B (~4.4GB)
8. ⏳ Falcon-7B (~4.2GB)
9. ⏳ ChatGLM-4-9B (~5.5GB) - if space allows
10. ⏳ Phi-2 (~1.6GB)

## Testing Protocol

For each model:
1. Download model to `~/.cache/models/gguf/`
2. Run shimmytok tokenization test
3. Run llama.cpp tokenization for reference
4. Compare outputs - must be 100% match
5. Mark as ✅ VALIDATED
6. Delete model file
7. Move to next

**Total Download Size**: ~35GB (sequential, not concurrent)
**Available Disk**: 13GB
**Strategy**: Download → Test → Delete → Repeat

## Current Status - Ready for Testing

**Downloaded Models (15.9GB total, disk 100% full):**

**Phase 1 (Core):**
- ✅ `gpt2.Q4_K_M.gguf` (108MB) - READY
- ✅ `qwen2-7b-instruct-q4_k_m.gguf` (4.4GB) - READY
- ✅ `starcoder2-3b-Q4_K_M.gguf` (1.8GB) - READY  
- ✅ `deepseek-coder-6.7b-instruct.Q4_K_M.gguf` (3.9GB) - READY

**Phase 2 (Secondary):**
- ✅ `deepseek-llm-7b-chat.Q4_K_M.gguf` (4.0GB) - READY
- ✅ `phi-2.Q4_K_M.gguf` (1.7GB) - READY

**Failed Downloads (not available in Q4_K_M):**
- ❌ BLOOM-7B - no Q4_K_M available
- ❌ Falcon-7B - no Q4_K_M available

**System Status:**
- Disk space remaining: 3.8GB (100% full)
- GPU VRAM: RTX 3060 12GB
- **Must delete models after testing to continue**

**Patterns Covered:**
- `gpt2` → 11 model variants
- `qwen2` → 5 model variants
- `starcoder` → 7 model variants
- `deepseek-coder` → 1 model
- `deepseek-llm` → 1 model
- `phi-2` → GPT-2 pattern validation

**Total**: 5 unique patterns covering 26 model variants

**Next Action**: User shuts down processes → Test all 6 models → Delete after validation
