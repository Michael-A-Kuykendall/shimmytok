# ShimmyTok — 純 Rust GGUF Tokenizer

**語言：** [English](../../README.md) · [简体中文](../zh-CN/README.md) · **繁體中文**

**目前版本：0.8.5。** ShimmyTok 是 Rust GGUF 分詞函式庫，並非命令列推論伺服器。能載入詞彙或產生 token **不表示** Airframe 已驗證完整神經網路推論。

## 功能與相容性界限

`Tokenizer::from_vocab` 支援 SentencePiece（llama/mistral/gemma/gemma4/gemma6）、BPE（gpt2/qwen/qwen2）、WordPiece（bert/wpm）、RWKV、Unigram（t5/ugm）與 PLaMo-2。不認識的 tokenizer model 必須回傳不支援錯誤。

介面包含 `encode`、`encode_with_options`、`decode`、`decode_single`、`encode_batch` 及唯讀 metadata。批次輸出保留輸入順序，錯誤遵循最小失敗索引。相容性限於實際驗證的模型、token ID、正規化和選項；不保證所有 llama.cpp 版本完全一致。

## 安裝與範例

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

請依照實際 GGUF chat template 設定特殊 token 選項，**避免重複新增 BOS**。明確的 GGUF 特殊 token 中繼資料優先於模型家族預設值；0.8.5 並未加入新的 BOS 預設修正。

## 安全、平台及授權

共用 GGUF 解析器會檢查陣列長度、配置預算、溢位、張量邊界與截斷，但不能保證任意不受信任模型檔均安全。執行證據來自 Linux/WSL；其他原生平台、WASM/WASI 及最低工具鏈認證仍為 NOT_RUN。

採 **MIT OR Apache-2.0** 雙授權，可選擇其中一項並保留聲明。參閱[英文 README](../../README.md)、[變更記錄](../../CHANGELOG.md)、[路線圖](../../ROADMAP.md)、[API 穩定性](../API_STABILITY.md)、[GGUF 安全說明](../GGUF_SECURITY.md)及[授權檔](../../LICENSE-MIT)。

英文文件是技術合約的權威版本。