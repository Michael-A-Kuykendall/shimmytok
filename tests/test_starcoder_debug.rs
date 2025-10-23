use fancy_regex::Regex;

#[test]
fn test_starcoder_sequential() {
    let text = "function main() { return 0; }";
    
    // StarCoder patterns
    let pattern1 = r"\p{N}";
    let pattern2 = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
    
    let regex1 = Regex::new(pattern1).unwrap();
    let regex2 = Regex::new(pattern2).unwrap();
    
    println!("\n=== Input: {:?} ===", text);
    
    // Apply first pattern
    println!("\n=== Pattern 1: \\p{{N}} (digits only) ===");
    let mut offsets: Vec<(usize, usize)> = vec![(0, text.len())];
    
    let mut new_offsets = Vec::new();
    for (start, end) in offsets {
        let fragment = &text[start..end];
        
        let matches: Vec<_> = regex1.find_iter(fragment).filter_map(|m| m.ok()).collect();
        
        if matches.is_empty() {
            println!("  No matches in fragment, keeping {}..{}", start, end);
            new_offsets.push((start, end));
        } else {
            let mut last_pos = 0;
            for m in matches {
                if m.start() > last_pos {
                    println!("  Gap: {:?} at {}..{}", &fragment[last_pos..m.start()], start + last_pos, start + m.start());
                    new_offsets.push((start + last_pos, start + m.start()));
                }
                println!("  Match: {:?} at {}..{}", m.as_str(), start + m.start(), start + m.end());
                new_offsets.push((start + m.start(), start + m.end()));
                last_pos = m.end();
            }
            if last_pos < fragment.len() {
                println!("  Final gap: {:?} at {}..{}", &fragment[last_pos..], start + last_pos, end);
                new_offsets.push((start + last_pos, end));
            }
        }
    }
    
    offsets = new_offsets;
    println!("\nOffsets after pattern 1: {:?}", offsets);
    println!("Fragments: {:?}", offsets.iter().map(|(s, e)| &text[*s..*e]).collect::<Vec<_>>());
    
    // Apply second pattern
    println!("\n=== Pattern 2: (main pattern) ===");
    new_offsets = Vec::new();
    for (start, end) in offsets {
        let fragment = &text[start..end];
        let mut has_matches = false;
        
        println!("\nProcessing fragment {:?} ({}..{})", fragment, start, end);
        for m in regex2.find_iter(fragment) {
            let m = m.unwrap();
            has_matches = true;
            println!("  Match: {:?} at {}..{}", m.as_str(), start + m.start(), start + m.end());
            new_offsets.push((start + m.start(), start + m.end()));
        }
        
        if !has_matches {
            println!("  No matches, keeping fragment");
            new_offsets.push((start, end));
        }
    }
    
    offsets = new_offsets;
    println!("\n=== Final offsets: {:?} ===", offsets);
    let fragments: Vec<String> = offsets.iter().map(|(s, e)| text[*s..*e].to_string()).collect();
    println!("Final fragments: {:?}", fragments);
    println!("Count: {}", fragments.len());
}
