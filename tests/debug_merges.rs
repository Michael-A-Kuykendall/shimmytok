use shimmytok::Tokenizer;

#[test]
#[ignore]
fn debug_gpt2_merges() {
    let model_path = std::env::var("HOME").unwrap() + "/.cache/models/gguf/gpt2.Q4_K_M.gguf";
    let tokenizer = Tokenizer::from_gguf_file(&model_path).unwrap();
    
    // Try to encode just "() {"
    let result = tokenizer.encode("() {", false).unwrap();
    println!("Tokens for '() {{': {:?}", result);
    
    // Expected from llama.cpp for "function main() { return 0; }":
    // [8818, 1388, 3419, 1391, 1441, 657, 26, 1782]
    
    // Try individual parts
    println!("'function': {:?}", tokenizer.encode("function", false).unwrap());
    println!("' main': {:?}", tokenizer.encode(" main", false).unwrap());
    println!("'()': {:?}", tokenizer.encode("()", false).unwrap());
    println!("' {{': {:?}", tokenizer.encode(" {", false).unwrap());
    println!("' return': {:?}", tokenizer.encode(" return", false).unwrap());
    println!("' 0': {:?}", tokenizer.encode(" 0", false).unwrap());
    println!("';': {:?}", tokenizer.encode(";", false).unwrap());
    println!("' }}': {:?}", tokenizer.encode(" }", false).unwrap());
}
