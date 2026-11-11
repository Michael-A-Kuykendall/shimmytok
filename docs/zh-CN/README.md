# ShimmyTok — 纯 Rust GGUF 分词器

**语言：** [English](../../README.md) · **简体中文** · [繁體中文](../zh-TW/README.md)

**当前版本：0.8.5。** ShimmyTok 是供 Rust 程序使用的 GGUF tokenizer 库，并非命令行推理服务器。能加载词表或分词**不代表**神经网络模型已通过 Airframe 推理认证。

## 功能与兼容边界

`Tokenizer::from_vocab` 分派 SentencePiece（llama/mistral/gemma/gemma4/gemma6）、BPE（gpt2/qwen/qwen2）、WordPiece（bert/wpm）、RWKV、Unigram（t5/ugm）以及 PLaMo-2。未识别的 tokenizer 模型应返回不支持错误。

提供 `encode`、`encode_with_options`、`decode`、`decode_single`、`encode_batch` 和只读 metadata。批处理保持输入顺序，错误选择最低失败索引。兼容性仅适用于明确测试的模型、token ID、规范化及选项；不保证与 llama.cpp 所有版本完全一致。

## 安装与示例

```toml
[dependencies]
shimmytok = "0.8.5"
```

```rust
use shimmytok::{EncodeOptions, Tokenizer};

fn main() -> Result<(), shimmytok::Error> {
    let tokenizer = Tokenizer::from_gguf_file("model.gguf")?;
    let ids = tokenizer.encode_with_options(
        "Hello, world!",
        &EncodeOptions::with_parse_special(true, true),
    )?;
    println!("{}", tokenizer.decode(&ids, true)?);
    Ok(())
}
```

根据真实 GGUF chat template 选择特殊 token 规则，**不要重复添加 BOS**。显式 GGUF 特殊 token 元数据优先于族默认值；0.8.5 并没有添加新的 BOS 默认修复。

## 安全性、平台与许可

共享 GGUF 读取器检查数组长度、预算、溢出、张量边界及截断。它不意味着不受信任模型文件绝对安全。独立套件测试已有 Linux/WSL 运行证据。后续引擎集成测试也在 Apple M1（Virtual）、macOS 15.8.2 上核对了特定 Qwen3/Llama token ID；独立完整的原生平台分词器测试矩阵、WASM/WASI 和最低工具链认证仍为 NOT_RUN。

采用 **MIT OR Apache-2.0** 双许可证，任选其中一个并保留相关声明。详见[英文 README](../../README.md)、[变更记录](../../CHANGELOG.md)、[路线图](../../ROADMAP.md)、[API 稳定性](../API_STABILITY.md)、[GGUF 安全说明](../GGUF_SECURITY.md)及[许可证](../../LICENSE-MIT)。

英文本是技术合同的权威版本。

默认的 parallel feature 使用 Rayon 进行批处理；default-features = false 使用顺序实现。完整示例、API 稳定性、实现范围、安全和开发信息请参阅以下英文权威章节。

[ShimmyTok 0.8.5](../../README.md#shimmytok-085) · [Implementation coverage](../../README.md#implementation-coverage) · [Installation](../../README.md#installation) · [Quick Start](../../README.md#quick-start) · [API and compatibility](../../README.md#api-and-compatibility) · [Security and platforms](../../README.md#security-and-platforms) · [License](../../README.md#license) · [Support](../../README.md#support)