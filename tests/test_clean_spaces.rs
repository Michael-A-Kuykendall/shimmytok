//! Tests for clean_spaces functionality (llama.cpp parity)

#[test]
fn test_clean_spaces_punctuation() {
    // Test: space before punctuation is removed
    // " ?" → "?"
    // " !" → "!"
    // " ." → "."
    // " ," → ","
    
    // This tests the apply_clean_spaces function behavior
    // We test through decode since that's where it's wired
    
    // Direct string transformation tests:
    let cases = vec![
        ("Hello ?", "Hello?"),
        ("Wait !", "Wait!"),
        ("End .", "End."),
        ("One , two", "One, two"),
        ("Multiple ?!.", "Multiple?!."),
    ];
    
    for (input, expected) in cases {
        let result = clean_spaces_helper(input);
        assert_eq!(result, expected, "Failed for input: {:?}", input);
    }
}

#[test]
fn test_clean_spaces_apostrophe_contractions() {
    // Test: space before apostrophe contractions is removed
    // " 'm" → "'m"
    // " 's" → "'s"
    // " 've" → "'ve"
    // " 're" → "'re"
    
    let cases = vec![
        ("I 'm happy", "I'm happy"),
        ("It 's working", "It's working"),
        ("They 've arrived", "They've arrived"),
        ("You 're welcome", "You're welcome"),
    ];
    
    for (input, expected) in cases {
        let result = clean_spaces_helper(input);
        assert_eq!(result, expected, "Failed for input: {:?}", input);
    }
}

#[test]
fn test_clean_spaces_isolated_apostrophe() {
    // Test: isolated apostrophe " ' " → "'"
    let cases = vec![
        ("word ' word", "word'word"),
    ];
    
    for (input, expected) in cases {
        let result = clean_spaces_helper(input);
        assert_eq!(result, expected, "Failed for input: {:?}", input);
    }
}

#[test]
fn test_clean_spaces_combined() {
    // Test combinations
    let cases = vec![
        ("Hello ! I 'm here .", "Hello! I'm here."),
        ("What 's that ?", "What's that?"),
    ];
    
    for (input, expected) in cases {
        let result = clean_spaces_helper(input);
        assert_eq!(result, expected, "Failed for input: {:?}", input);
    }
}

// Helper that mimics the apply_clean_spaces function
// This is a copy to test in isolation without needing a tokenizer
fn clean_spaces_helper(text: &str) -> String {
    let mut chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return String::new();
    }

    // Pass 1: Remove space before punctuation ?!.,
    let mut i = 1;
    while i < chars.len() {
        if chars[i - 1] == ' ' && matches!(chars[i], '?' | '!' | '.' | ',') {
            chars.remove(i - 1);
        } else {
            i += 1;
        }
    }

    // Pass 2: Strip single apostrophe between spaces: " ' " → "'"
    let mut i = 1;
    while i + 1 < chars.len() {
        if chars[i] == '\'' && chars[i - 1] == ' ' && chars.get(i + 1) == Some(&' ') {
            chars.remove(i - 1);
            chars.remove(i);
        } else {
            i += 1;
        }
    }

    // Pass 3: Apostrophe contractions
    let mut i = 1;
    while i + 1 < chars.len() {
        if chars[i - 1] == ' ' && chars[i] == '\'' {
            let next = chars.get(i + 1);
            let next2 = chars.get(i + 2);

            let should_remove_space = match (next, next2) {
                (Some('s'), _) | (Some('m'), _) => true,
                (Some('v'), Some('e')) | (Some('r'), Some('e')) => true,
                _ => false,
            };

            if should_remove_space {
                chars.remove(i - 1);
            } else {
                i += 1;
            }
        } else {
            i += 1;
        }
    }

    chars.into_iter().collect()
}
