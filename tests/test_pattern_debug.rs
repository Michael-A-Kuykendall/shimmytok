use fancy_regex::Regex;

#[test]
fn test_gpt2_pattern_multiline() {
    let pattern = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
    let regex = Regex::new(pattern).unwrap();
    let text = "Multiple\nlines\nof\ntext";
    
    println!("Input: {:?}", text);
    println!("Pattern: {}", pattern);
    
    for m in regex.find_iter(text) {
        let m = m.unwrap();
        println!("  Match: {:?} at {}..{}", m.as_str(), m.start(), m.end());
    }
    
    let matches: Vec<_> = regex
        .find_iter(text)
        .filter_map(|m| m.ok())
        .map(|m| m.as_str())
        .collect();
    
    println!("Total matches: {:?}", matches);
    
    // The pattern should match "Multiple", "\n", "lines", "\n", "of", "\n", "text"
    assert_eq!(matches.len(), 7, "Should get 7 fragments");
}
