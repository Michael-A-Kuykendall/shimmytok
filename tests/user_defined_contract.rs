use shimmytok::{EncodeOptions, Tokenizer};
fn string(b: &mut Vec<u8>, s: &str) {
    b.extend((s.len() as u64).to_le_bytes());
    b.extend(s.as_bytes());
}
fn fixture() -> Tokenizer {
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend(0u64.to_le_bytes());
    b.extend(4u64.to_le_bytes());
    string(&mut b, "tokenizer.ggml.tokens");
    b.extend(9u32.to_le_bytes());
    b.extend(8u32.to_le_bytes());
    b.extend(5u64.to_le_bytes());
    for s in ["<unk>", "<tag>", "<tag_long>", "<ctrl>", "<normal>"] {
        string(&mut b, s);
    }
    string(&mut b, "tokenizer.ggml.token_type");
    b.extend(9u32.to_le_bytes());
    b.extend(5u32.to_le_bytes());
    b.extend(5u64.to_le_bytes());
    for t in [2i32, 4, 4, 3, 1] {
        b.extend(t.to_le_bytes());
    }
    for key in [
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.add_eos_token",
    ] {
        string(&mut b, key);
        b.extend(7u32.to_le_bytes());
        b.push(0);
    }
    Tokenizer::from_bytes(&b).unwrap()
}
#[test]
fn user_defined_is_atomic_with_both_parse_flags() {
    let t = fixture();
    for parse_special in [false, true] {
        let options = EncodeOptions {
            add_special_tokens: false,
            parse_special,
        };
        assert_eq!(
            t.encode_with_options("<tag><tag_long><tag>", &options)
                .unwrap(),
            [1, 2, 1]
        );
    }
}
#[test]
fn control_tokens_remain_opt_in() {
    let t = fixture();
    assert_eq!(
        t.encode_with_options(
            "<ctrl>",
            &EncodeOptions {
                add_special_tokens: false,
                parse_special: true
            }
        )
        .unwrap(),
        [3]
    );
    assert_ne!(
        t.encode_with_options(
            "<ctrl>",
            &EncodeOptions {
                add_special_tokens: false,
                parse_special: false
            }
        )
        .unwrap(),
        [3]
    );
}
#[test]
#[ignore = "requires exact Qwen3 fixture in SHIMMY_CERT_MODEL_DIR"]
fn exact_qwen_think_markers_match_llama_tokenizer() {
    let path = std::path::Path::new(&std::env::var("SHIMMY_CERT_MODEL_DIR").expect("fixtures"))
        .join("Qwen3-0.6B-Q4_K_M.gguf");
    let t = Tokenizer::from_gguf_file(&path).unwrap();
    for parse_special in [true, false] {
        let opts = EncodeOptions {
            add_special_tokens: true,
            parse_special,
        };
        assert_eq!(
            t.encode_with_options("<think>\n\n</think>\n\n", &opts)
                .unwrap(),
            [151667, 271, 151668, 271]
        );
    }
}

#[test]
#[ignore = "requires exact model and LLAMA_TOKENIZE_BIN pinned oracle"]
fn chat_tokens_match_independent_llama_cpp_for_both_parse_modes() {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let model = std::path::Path::new(&std::env::var("SHIMMY_CERT_MODEL_DIR").expect("fixtures"))
        .join("Qwen3-0.6B-Q4_K_M.gguf");
    let oracle = std::env::var("LLAMA_TOKENIZE_BIN").expect("pinned oracle");
    let tokenizer = Tokenizer::from_gguf_file(&model).unwrap();
    for prompt in [
  "<think>\n\n</think>\n\n",
  "<|im_start|>user\nWhat is 2 + 2? Answer briefly.<|im_end|>\n<|im_start|>assistant\n",
  "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nMy name is Ada.<|im_end|>\n<|im_start|>assistant\nHello Ada!<|im_end|>\n<|im_start|>user\nWhat is my name?\n/nothink<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
  "<|im_start|>system\nYou are a helpful assistant.\n/nothink<|im_end|>\n<|im_start|>user\nMy name is Ada.\n/nothink<|im_end|>\n<|im_start|>assistant\nHello Ada!<|im_end|>\n<|im_start|>user\nWhat is my name?\n/nothink<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
 ] {
  for parse_special in [false,true] {
   let mut command=Command::new(&oracle);command.arg("-m").arg(&model).args(["--stdin","--ids"]);
   if !parse_special {command.arg("--no-parse-special");}
   let mut child=command.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
   child.stdin.take().unwrap().write_all(prompt.as_bytes()).unwrap();
   let output=child.wait_with_output().unwrap();assert!(output.status.success(),"{}",String::from_utf8_lossy(&output.stderr));
   let expected:Vec<u32>=serde_json::from_slice(&output.stdout).unwrap();
   assert_eq!(tokenizer.encode_with_options(prompt,&EncodeOptions{add_special_tokens:true,parse_special}).unwrap(),expected,"parse_special={parse_special} prompt={prompt:?}");
  }
 }
}
