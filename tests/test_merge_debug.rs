use shimmytok::vocab::Vocabulary;

#[test]
fn test_merge_sequence() {
    let path = "../aistatepilot-mcp/models/Phi-3-mini-4k-instruct-q4.gguf";

    if std::path::Path::new(path).exists() {
        let vocab = Vocabulary::from_gguf_file(path).unwrap();

        // Simulate what should happen
        let text = "Hello world";
        let processed = format!("▁{}", text.replace(' ', "▁"));
        println!("Processed text: '{}'", processed);

        // Characters after splitting
        let chars: Vec<String> = processed.chars().map(|c| c.to_string()).collect();
        println!("Initial chars: {:?}", chars);

        // Check all possible bigrams and their scores
        println!("\nPossible bigrams from initial state:");
        for i in 0..chars.len() - 1 {
            let combined = format!("{}{}", chars[i], chars[i + 1]);
            if let Some(id) = vocab.get_token_id(&combined) {
                let score = vocab.get_token_score(id);
                println!(
                    "  '{}' + '{}' = '{}' (token {}, score {})",
                    chars[i],
                    chars[i + 1],
                    combined,
                    id,
                    score
                );
            }
        }

        // What should happen:
        // 1. ▁ + H should merge? Check token
        if let Some(id) = vocab.get_token_id("▁H") {
            println!(
                "\n'▁H' exists as token {} with score {}",
                id,
                vocab.get_token_score(id)
            );
        }

        // After first best merge, what next?
        // Should eventually get ▁Hello

        // Critical question: after we have ▁Hello, what happens to ▁world?
        println!("\nAfter getting ▁Hello, remaining chars would be: ▁, w, o, r, l, d");

        // Can we merge ▁ + w?
        if let Some(id) = vocab.get_token_id("▁w") {
            println!(
                "'▁w' exists as token {} with score {}",
                id,
                vocab.get_token_score(id)
            );
        }

        // After ▁w, can we continue to get ▁world?
        // This is where the algorithm might be failing

        println!("\nCritical check: After merging to '▁wo', can we still get '▁world'?");
        println!("In llama.cpp, the resegment function would check this AFTER merging completes");

        // The real issue: we need to check all possible segmentations
        // Not just greedy merging
    }
}
