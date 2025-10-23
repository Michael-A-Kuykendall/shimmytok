use fancy_regex::Regex;

fn main() {
    let pattern = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)";
    let regex = Regex::new(pattern).unwrap();
    let text = "Multiple\nlines\nof\ntext";
    
    println!("Input: {:?}", text);
    println!("Pattern: {}", pattern);
    println!("\nMatches:");
    
    for m in regex.find_iter(text) {
        let m = m.unwrap();
        println!("  {:?} (bytes {}..{})", m.as_str(), m.start(), m.end());
    }
}
