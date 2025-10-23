#!/bin/bash
# BPE Model Download Script
# Downloads models sequentially to avoid disk space issues

set -e

CACHE_DIR="$HOME/.cache/models/gguf"
mkdir -p "$CACHE_DIR"
cd "$CACHE_DIR"

echo "=== BPE Model Downloads ==="
echo "Available disk space:"
df -h . | tail -1

echo ""
echo "Phase 1: Core Patterns"
echo "======================"

# 1. GPT-2 (already have)
if [ -f "gpt2.Q4_K_M.gguf" ]; then
    echo "✅ GPT-2 model already present"
else
    echo "⚠️  GPT-2 model missing - download separately"
fi

# 2. Qwen2-7B
if [ -f "qwen2-7b-instruct-q4_k_m.gguf" ]; then
    echo "✅ Qwen2-7B already downloaded"
else
    echo "📥 Downloading Qwen2-7B Q4_K_M (~4.4GB)..."
    curl -L -O "https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF/resolve/main/qwen2-7b-instruct-q4_k_m.gguf"
fi

# 3. StarCoder-3B  
echo ""
echo "📥 Downloading StarCoder-3B Q4_K_M (~2GB)..."
if [ ! -f "starcoder2-3b-Q4_K_M.gguf" ]; then
    curl -L -O "https://huggingface.co/second-state/StarCoder2-3B-GGUF/resolve/main/starcoder2-3b-Q4_K_M.gguf"
else
    echo "✅ StarCoder already downloaded"
fi

# 4. Llama-3-8B BPE (need to find correct variant)
echo ""
echo "⚠️  Llama-3-8B BPE: Need to find BPE variant (not SentencePiece)"
echo "    Skipping for now - will research correct model"

echo ""
echo "Phase 2: Important Secondary"
echo "============================"

# 5. DeepSeek-Coder-6.7B
echo "📥 Downloading DeepSeek-Coder-6.7B Q4_K_M (~4GB)..."
if [ ! -f "deepseek-coder-6.7b-instruct.Q4_K_M.gguf" ]; then
    curl -L -O "https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF/resolve/main/deepseek-coder-6.7b-instruct.Q4_K_M.gguf"
else
    echo "✅ DeepSeek-Coder already downloaded"
fi

# 6. DeepSeek-LLM-7B
echo ""
echo "📥 Downloading DeepSeek-LLM-7B Q4_K_M (~4.4GB)..."
if [ ! -f "deepseek-llm-7b-chat.Q4_K_M.gguf" ]; then
    curl -L -O "https://huggingface.co/TheBloke/deepseek-llm-7B-chat-GGUF/resolve/main/deepseek-llm-7b-chat.Q4_K_M.gguf"
else
    echo "✅ DeepSeek-LLM already downloaded"
fi

# 7. BLOOM-7B
echo ""
echo "📥 Downloading BLOOM-7B Q4_K_M (~4.4GB)..."
if [ ! -f "bloomz-7b1-q4_k_m.gguf" ]; then
    curl -L -O "https://huggingface.co/TheBloke/bloomz-7B1-GGUF/resolve/main/bloomz-7b1.Q4_K_M.gguf"
else
    echo "✅ BLOOM already downloaded"
fi

echo ""
echo "Phase 3: Optional"
echo "================"

# 8. Falcon-7B
echo "📥 Downloading Falcon-7B Q4_K_M (~4.2GB)..."
if [ ! -f "falcon-7b.Q4_K_M.gguf" ]; then
    curl -L -O "https://huggingface.co/TheBloke/falcon-7b-GGUF/resolve/main/falcon-7b.Q4_K_M.gguf"
else
    echo "✅ Falcon already downloaded"
fi

# 9. Phi-2
echo ""
echo "📥 Downloading Phi-2 Q4_K_M (~1.6GB)..."
if [ ! -f "phi-2.Q4_K_M.gguf" ]; then
    curl -L -O "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"
else
    echo "✅ Phi-2 already downloaded"
fi

echo ""
echo "=== Download Summary ==="
ls -lh *.gguf | awk '{print $9, $5}'
echo ""
df -h . | tail -1
